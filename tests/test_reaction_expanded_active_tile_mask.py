from __future__ import annotations

import inspect
from types import SimpleNamespace

from oracle_game.sim import gpu_reactions_cell_pass, gpu_reactions_pairings
from oracle_game.sim.gpu_reactions import GPUReactionPipeline, GPUReactionResources, _SHADER_SUBS
from oracle_game.sim.shader_loader import shader_source


def test_reaction_expanded_active_tile_mask_is_default_enabled() -> None:
    pipeline = GPUReactionPipeline()

    assert pipeline._expanded_active_tile_mask_enabled is True
    assert pipeline.last_expanded_active_tile_mask_used is False
    assert pipeline.expanded_active_tile_mask_build_count == 0
    assert "expanded_active_tile_tex" in GPUReactionResources.__dataclass_fields__


def test_reaction_expanded_active_tile_builder_is_r8_and_exact_3x3_ttl_expansion() -> None:
    source = shader_source(
        "reactions/load_expanded_active_tiles.comp",
        _SHADER_SUBS,
        includes=["reactions/_active_helper.comp"],
    )

    assert "layout(r8ui, binding=1) writeonly uniform uimage2D" in source
    assert "source_tile_active(ivec2(source_x, source_y))" in source
    assert "tile.y - expansion_radius" in source
    assert "tile.y + expansion_radius" in source
    assert "tile.x - expansion_radius" in source
    assert "tile.x + expansion_radius" in source
    assert "expanded_tile_active(tile) ? 1u : 0u" in source


def test_reaction_expanded_active_tile_solve_has_no_shared_barrier() -> None:
    for common_name in ("_common.comp", "_common_self_apply.comp"):
        source = shader_source(f"reactions/{common_name}", _SHADER_SUBS)
        assert "uniform usampler2D expanded_active_tile_tex" in source
        assert "texelFetch(expanded_active_tile_tex, cell >> 5, 0).x != 0u" in source
        assert "barrier();" not in source


def test_reaction_expanded_active_tile_gate_is_formal_tile32_and_sparse_falls_back() -> None:
    pipeline = GPUReactionPipeline()
    pipeline._expanded_active_tile_mask_enabled = True
    world = SimpleNamespace(
        active=SimpleNamespace(tile_size=32),
        simulation_backend="gpu",
        _world_simulation_frame_active=True,
        bridge=SimpleNamespace(gpu_authoritative_resources={"active_tile_ttl"}),
    )

    assert pipeline._can_use_expanded_active_tile_mask(world) is True
    world.active.tile_size = 16
    assert pipeline._can_use_expanded_active_tile_mask(world) is False
    world.active.tile_size = 32
    world._world_simulation_frame_active = False
    assert pipeline._can_use_expanded_active_tile_mask(world) is False

    timed_source = inspect.getsource(gpu_reactions_pairings.run_timed_actions)
    self_source = inspect.getsource(gpu_reactions_pairings.run_self_actions)
    triplet_source = inspect.getsource(gpu_reactions_cell_pass._run_material_pair_fused_pass)
    assert 'not getattr(pipeline, "_timed_sparse_inplace_enabled", False)' in timed_source
    assert "and not sparse_inplace" in self_source
    assert "terminal_handoff" in triplet_source
    assert "use_expanded_tile_mask=use_expanded_active_tile_mask" in triplet_source


def test_reaction_active_mask_cache_tracks_cell_tile_and_gas_representations() -> None:
    pipeline = GPUReactionPipeline()
    pipeline._expanded_active_tile_mask_enabled = True
    world = SimpleNamespace(
        frame_id=7,
        simulation_backend="gpu",
        _world_simulation_frame_active=True,
        active=SimpleNamespace(tile_size=32),
        bridge=SimpleNamespace(
            gpu_authoritative_resources={"active_tile_ttl"},
            buffers={
                "active_tile_ttl": object(),
                "active_chunk_mask": object(),
                "active_meta": object(),
            },
        ),
    )
    resources = SimpleNamespace(signature=(67, 67, 17, 17, 6, 4))
    pipeline._formal_segment_batch_base_key = (id(world), 7, "before_motion")

    cell = pipeline._formal_reaction_active_mask_cache_key(
        world,
        resources,
        "timed",
        expansion_radius=1,
        load_cell_mask=True,
        load_expanded_tile_mask=False,
        load_gas_mask=False,
    )
    tile = pipeline._formal_reaction_active_mask_cache_key(
        world,
        resources,
        "timed",
        expansion_radius=1,
        load_cell_mask=False,
        load_expanded_tile_mask=True,
        load_gas_mask=False,
    )
    tile_gas = pipeline._formal_reaction_active_mask_cache_key(
        world,
        resources,
        "self",
        expansion_radius=1,
        load_cell_mask=False,
        load_expanded_tile_mask=True,
        load_gas_mask=True,
    )

    assert cell is not None and tile is not None and tile_gas is not None
    assert cell[:-3] == tile[:-3] == tile_gas[:-3]
    assert cell[-3:] == (True, False, False)
    assert tile[-3:] == (False, True, False)
    assert tile_gas[-3:] == (False, True, True)


def test_reaction_gas_only_passes_upload_only_the_canonical_gas_mask() -> None:
    for runner in (
        gpu_reactions_pairings.run_gas_gas,
        gpu_reactions_pairings.run_gas_light,
        gpu_reactions_pairings._run_formal_guarded_gas_light,
    ):
        source = inspect.getsource(runner)
        assert "load_cell_mask=False" in source
        assert "load_gas_mask=False" not in source
