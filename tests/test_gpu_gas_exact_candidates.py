from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from oracle_game.sim.gpu_gas import GPUGasPipeline
from oracle_game.types import ForceSource
from oracle_game.world import WorldEngine


def test_gpu_gas_exact_candidates_are_default_enabled() -> None:
    pipeline = GPUGasPipeline()
    assert pipeline._divergence_pressure_seed_enabled is True
    assert pipeline._species_terminal_cooperative_enabled is True
    assert pipeline._density_tree_reduction_enabled is True


def _add_species_to_capacity(engine: WorldEngine) -> None:
    water_id = engine.rulebook.gas_id("water_gas")
    template = engine.rulebook.gases_by_id[water_id]
    additions = [
        replace(
            template,
            species_id=6,
            name="exact_test_gas_6",
            display_name="Exact Test Gas 6",
            diffusion_rate=0.037,
            buoyancy=-0.19,
            decay_rate=0.013,
            temperature_coupling=0.17,
            pressure_factor=0.83,
            density_factor=1.31,
            condense_point=None,
            condense_to_material=None,
        ),
        replace(
            template,
            species_id=7,
            name="exact_test_gas_7",
            display_name="Exact Test Gas 7",
            diffusion_rate=0.091,
            buoyancy=0.27,
            decay_rate=0.029,
            temperature_coupling=-0.08,
            pressure_factor=1.17,
            density_factor=0.72,
            condense_point=None,
            condense_to_material=None,
        ),
    ]
    engine.update_gas_species_table(additions)


def _seed_gas_world(engine: WorldEngine, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    engine.flow_velocity[:] = rng.uniform(-1.75, 1.75, size=engine.flow_velocity.shape).astype("f4")
    engine.ambient_temperature[:] = rng.uniform(-80.0, 240.0, size=engine.ambient_temperature.shape).astype("f4")
    engine.gas_concentration[:] = rng.uniform(0.0, 1.4, size=engine.gas_concentration.shape).astype("f4")
    engine.pressure_ping[:] = rng.uniform(-3.0, 3.0, size=engine.pressure_ping.shape).astype("f4")
    engine.force_sources.append(
        ForceSource(
            x=23.5,
            y=17.25,
            direction=(0.75, -0.4),
            radius=13.0,
            strength=1.8,
            lifetime=0.7,
        )
    )
    solve_mask = rng.random((engine.gas_height, engine.gas_width)) > 0.37
    solve_mask[0, 0] = True
    solve_mask[0, -1] = False
    solve_mask[-1, 0] = False
    solve_mask[-1, -1] = True
    return solve_mask


def _capture_gas_state(engine: WorldEngine) -> dict[str, bytes]:
    pipeline = engine.gas_solver.gpu_pipeline
    resources = pipeline.resources
    ctx = engine.bridge.ctx
    assert resources is not None and ctx is not None
    ctx.finish()
    return {
        "resident.velocity_ping": resources.velocity_ping.read(),
        "resident.velocity_pong": resources.velocity_pong.read(),
        "resident.divergence": resources.divergence.read(),
        "resident.thermo_pressure": resources.thermo_pressure.read(),
        "resident.density": resources.density_tex.read(),
        "resident.pressure_ping": resources.pressure_ping.read(),
        "resident.pressure_pong": resources.pressure_pong.read(),
        "resident.ambient_ping": resources.ambient_ping.read(),
        "resident.ambient_pong": resources.ambient_pong.read(),
        "resident.gas_ping": resources.gas_ping.read(),
        "resident.gas_pong": resources.gas_pong.read(),
        "resident.active_gas": resources.active_gas_tex.read(),
        "bridge.velocity": engine.bridge.textures["flow_velocity"].read(),
        "bridge.ambient": engine.bridge.textures["ambient_temperature"].read(),
        "bridge.pressure": engine.bridge.textures["pressure_ping"].read(),
        "bridge.gas": engine.bridge.buffers["gas_concentration"].read(size=engine.gas_concentration.nbytes),
        "cpu.velocity": engine.flow_velocity.tobytes(),
        "cpu.ambient": engine.ambient_temperature.tobytes(),
        "cpu.pressure": engine.pressure_ping.tobytes(),
        "cpu.gas": engine.gas_concentration.tobytes(),
    }


def _run_gas_candidate(
    *,
    species_count: int,
    pressure_seed: bool,
    cooperative_terminal: bool,
    density_tree: bool | None = None,
) -> tuple[list[dict[str, bytes]], tuple[bool, bool]]:
    # 67x45 becomes a 17x12 gas grid. Both candidate workgroup axes therefore
    # exercise partial right and bottom workgroups in addition to all corners.
    engine = WorldEngine(width=67, height=45, gas_cell_size=4)
    try:
        pipeline = engine.gas_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU gas pipeline is not available")
        if species_count == 8:
            _add_species_to_capacity(engine)
        assert int(engine.gas_concentration.shape[0]) == species_count
        solve_mask = _seed_gas_world(engine, seed=0x6A51 + species_count)
        pipeline._divergence_pressure_seed_enabled = pressure_seed
        pipeline._species_terminal_cooperative_enabled = cooperative_terminal
        if density_tree is not None:
            pipeline._density_tree_reduction_enabled = density_tree

        ctx = engine.bridge.ctx
        assert ctx is not None
        pipeline._ensure_programs(ctx)
        resources = pipeline._ensure_resources(engine)
        poison = np.full(
            (engine.gas_height, engine.gas_width),
            np.float32(12345.75),
            dtype=np.float32,
        ).tobytes()
        resources.pressure_ping.write(poison)
        resources.pressure_pong.write(poison)

        snapshots: list[dict[str, bytes]] = []
        pipeline.step(engine, 1.0 / 60.0, solve_gas_mask=solve_mask)
        snapshots.append(_capture_gas_state(engine))

        # Keep the resident allocation alive while invalidating the cached
        # parameter signature, then exercise a second role swap and mask.
        patch_name = "exact_test_gas_7" if species_count == 8 else "water_gas"
        engine.patch_gas(
            patch_name,
            diffusion_rate=0.143,
            buoyancy=-0.33,
            decay_rate=0.047,
            temperature_coupling=0.21,
            pressure_factor=1.29,
            density_factor=0.68,
        )
        pipeline.step(
            engine,
            1.0 / 55.0,
            solve_gas_mask=np.roll(solve_mask, shift=(1, -2), axis=(0, 1)),
        )
        snapshots.append(_capture_gas_state(engine))
        used = (
            pipeline.last_divergence_pressure_seed_used,
            pipeline.last_species_terminal_cooperative_used,
        )
        return snapshots, used
    finally:
        engine.close()


def _run_formal_gas_candidate(
    *,
    species_count: int,
    pressure_seed: bool,
    cooperative_terminal: bool,
    density_tree: bool | None = None,
) -> tuple[dict[str, bytes], tuple[bool, bool]]:
    engine = WorldEngine(width=67, height=45, gas_cell_size=4)
    try:
        pipeline = engine.gas_solver.gpu_pipeline
        if not pipeline.available(engine):
            pytest.skip("GPU gas pipeline is not available")
        if species_count == 8:
            _add_species_to_capacity(engine)
        solve_mask = _seed_gas_world(engine, seed=0xF041 + species_count)
        for tile_y in range(engine.active.tile_height):
            for tile_x in range(engine.active.tile_width):
                engine.active.active_tile_ttl[tile_y][tile_x] = (
                    3 if (tile_x + 2 * tile_y) % 3 else 0
                )
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine.bridge.mark_gpu_authoritative(
            "flow_velocity",
            "ambient_temperature",
            "gas_concentration",
            "active_meta",
            "active_tile_ttl",
            "active_chunk_mask",
        )
        pipeline._divergence_pressure_seed_enabled = pressure_seed
        pipeline._species_terminal_cooperative_enabled = cooperative_terminal
        if density_tree is not None:
            pipeline._density_tree_reduction_enabled = density_tree
        ctx = engine.bridge.ctx
        assert ctx is not None
        pipeline._ensure_programs(ctx)
        resources = pipeline._ensure_resources(engine)
        poison = np.full(
            (engine.gas_height, engine.gas_width),
            np.float32(-9876.5),
            dtype=np.float32,
        ).tobytes()
        resources.pressure_ping.write(poison)
        resources.pressure_pong.write(poison)

        previous_frame_active = engine._world_simulation_frame_active
        engine._world_simulation_frame_active = True
        try:
            pipeline.step(engine, 1.0 / 60.0, solve_gas_mask=solve_mask)
        finally:
            engine._world_simulation_frame_active = previous_frame_active
        return _capture_gas_state(engine), (
            pipeline.last_divergence_pressure_seed_used,
            pipeline.last_species_terminal_cooperative_used,
        )
    finally:
        engine.close()


@pytest.mark.parametrize("species_count", (6, 8))
@pytest.mark.parametrize(
    ("pressure_seed", "cooperative_terminal"),
    ((True, False), (False, True), (True, True)),
    ids=("pressure-seed", "cooperative-terminal", "combined"),
)
def test_gpu_gas_candidates_match_legacy_raw_state_exactly(
    species_count: int,
    pressure_seed: bool,
    cooperative_terminal: bool,
) -> None:
    expected, control_used = _run_gas_candidate(
        species_count=species_count,
        pressure_seed=False,
        cooperative_terminal=False,
    )
    actual, candidate_used = _run_gas_candidate(
        species_count=species_count,
        pressure_seed=pressure_seed,
        cooperative_terminal=cooperative_terminal,
    )
    assert control_used == (False, False)
    assert candidate_used == (pressure_seed, cooperative_terminal)
    assert len(actual) == len(expected) == 2
    for frame_index, (expected_frame, actual_frame) in enumerate(zip(expected, actual, strict=True), start=1):
        assert actual_frame.keys() == expected_frame.keys()
        for resource_name in expected_frame:
            assert actual_frame[resource_name] == expected_frame[resource_name], (
                f"frame {frame_index}, {species_count} species A={pressure_seed} "
                f"B={cooperative_terminal} differs in {resource_name}"
            )


@pytest.mark.parametrize("species_count", (6, 8))
@pytest.mark.parametrize(
    ("pressure_seed", "cooperative_terminal"),
    ((True, False), (False, True), (True, True)),
    ids=("pressure-seed", "cooperative-terminal", "combined"),
)
def test_gpu_gas_formal_candidates_match_legacy_bridge_and_resident_state_exactly(
    species_count: int,
    pressure_seed: bool,
    cooperative_terminal: bool,
) -> None:
    expected, control_used = _run_formal_gas_candidate(
        species_count=species_count,
        pressure_seed=False,
        cooperative_terminal=False,
    )
    actual, candidate_used = _run_formal_gas_candidate(
        species_count=species_count,
        pressure_seed=pressure_seed,
        cooperative_terminal=cooperative_terminal,
    )
    assert control_used == (False, False)
    assert candidate_used == (pressure_seed, cooperative_terminal)
    assert actual.keys() == expected.keys()
    for resource_name in expected:
        assert actual[resource_name] == expected[resource_name], (
            f"formal {species_count} species A={pressure_seed} B={cooperative_terminal} "
            f"differs in {resource_name}"
        )


def test_gpu_gas_cooperative_terminal_falls_back_above_shader_capacity() -> None:
    engine = WorldEngine(width=32, height=24, gas_cell_size=4)
    try:
        pipeline = engine.gas_solver.gpu_pipeline
        pipeline._species_terminal_cooperative_enabled = True
        engine.gas_concentration = np.zeros(
            (9, engine.gas_height, engine.gas_width),
            dtype=np.float32,
        )
        assert pipeline._can_run_species_terminal_cooperative(engine) is False
    finally:
        engine.close()


@pytest.mark.parametrize("species_count", (6, 8))
def test_gpu_gas_density_tree_matches_pairwise_reduction_raw_state_exactly(
    species_count: int,
) -> None:
    expected_frames, _ = _run_gas_candidate(
        species_count=species_count,
        pressure_seed=True,
        cooperative_terminal=True,
        density_tree=False,
    )
    actual_frames, _ = _run_gas_candidate(
        species_count=species_count,
        pressure_seed=True,
        cooperative_terminal=True,
        density_tree=True,
    )
    for frame_index, (expected, actual) in enumerate(
        zip(expected_frames, actual_frames, strict=True), start=1
    ):
        for resource_name in expected:
            assert actual[resource_name] == expected[resource_name], (
                f"density tree species={species_count} frame={frame_index} "
                f"resource={resource_name} differs"
            )

    expected_formal, _ = _run_formal_gas_candidate(
        species_count=species_count,
        pressure_seed=True,
        cooperative_terminal=True,
        density_tree=False,
    )
    actual_formal, _ = _run_formal_gas_candidate(
        species_count=species_count,
        pressure_seed=True,
        cooperative_terminal=True,
        density_tree=True,
    )
    for resource_name in expected_formal:
        assert actual_formal[resource_name] == expected_formal[resource_name], (
            f"formal density tree species={species_count} resource={resource_name} differs"
        )
