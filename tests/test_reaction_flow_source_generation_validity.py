from __future__ import annotations

import inspect
import re

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.gpu_reactions_bridge import _apply_flow_sources_to_bridge_velocity
from oracle_game.sim.gpu_reactions_resources import _ensure_resources
from oracle_game.sim.gpu_reactions_transient import (
    _advance_flow_source_generation,
    _bind_flow_source_generation_output,
    _clear_transient_state,
)
from oracle_game.sim.gpu_reactions_cell_pass import _run_timed_candidate_action_pass
from oracle_game.sim.gpu_reactions_pairings import run_self_actions, run_timed_actions
from oracle_game.sim.gpu_reactions_side_effects import (
    _run_cell_gas_action_delta_pass,
    _run_self_candidate_gas_side_effect_pass,
)
from oracle_game.sim.shader_loader import shader_source


FLOW_SOURCE_PRODUCERS = (
    "reactions/scatter_cell_gas_action_delta.comp",
    "reactions/scatter_cell_gas_action_delta_candidates.comp",
    "reactions/scatter_self_gas_action_delta_candidates.comp",
    "reactions/cell_gas_side_effects.comp",
    "reactions/gas_gas.comp",
    "reactions/gas_light.comp",
)


def test_flow_source_generation_validity_is_default_for_formal_programs() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._flow_source_generation_validity_enabled is True
    assert pipeline._flow_source_generation_programs_enabled is False
    assert pipeline._flow_source_generation_u8_token_enabled is True
    assert pipeline._flow_source_generation_u8_programs_enabled is False
    assert _SHADER_SUBS["FLOW_SOURCE_GENERATION_VALIDITY"] == 0

    clear_source = inspect.getsource(_clear_transient_state)
    assert "pipeline._flow_source_generation_validity_active(world)" in clear_source
    assert "pipeline._advance_flow_source_generation(world, resources)" in clear_source
    assert 'pipeline.programs["clear_transient_flow_sources"]' in clear_source

    ensure_source = inspect.getsource(GPUReactionPipeline._ensure_programs)
    assert "self._flow_source_generation_programs_enabled = bool(" in ensure_source
    assert '"FLOW_SOURCE_GENERATION_VALIDITY": int(' in ensure_source
    assert ensure_source.count("flow_source_subs") >= 9


def test_flow_source_generation_resource_is_zero_initialized_and_wrap_cleared() -> None:
    resource_source = inspect.getsource(_ensure_resources)
    assert "flow_source_generation_tex=ctx.texture_array" in resource_source
    assert "dtype=np.uint32" in resource_source

    advance_source = inspect.getsource(_advance_flow_source_generation)
    assert "np.iinfo(np.uint32).max" in advance_source
    assert 'pipeline.programs["clear_transient_flow_source_generations"]' in advance_source
    assert "resources.flow_source_generation = generation + 1" in advance_source

    shader = shader_source(
        "reactions/clear_transient_flow_source_generations.comp",
        _SHADER_SUBS,
    )
    assert "layout(r32ui, binding=0)" in shader
    assert "imageStore(flow_source_generation_img, gid, uvec4(0u));" in shader


def test_every_flow_source_producer_publishes_generation_after_payload() -> None:
    for shader_name in FLOW_SOURCE_PRODUCERS:
        source = shader_source(shader_name)
        assert "uniform bool flow_source_generation_validity_enabled;" in source
        assert "uniform uint flow_source_generation;" in source
        assert "writeonly uniform uimage2DArray flow_source_generation_img;" in source
        payload_store = source.index("imageStore(flow_source_img, location, source);")
        generation_store = source.index(
            "imageStore(flow_source_generation_img, location, uvec4(flow_source_generation));"
        )
        assert payload_store < generation_store
        assert source.count("imageStore(flow_source_img") == 1

    fused = shader_source("reactions/_self_fused_gas_output.comp")
    payload_store = fused.index("imageStore(self_fused_flow_source_img, location, source);")
    generation_store = fused.index(
        "imageStore(flow_source_generation_img, location, uvec4(flow_source_generation));"
    )
    assert payload_store < generation_store
    assert fused.count("imageStore(self_fused_flow_source_img") == 1


def test_default_flow_source_program_specialization_compiles_generation_state_out() -> None:
    def strip_disabled_blocks(source: str) -> str:
        return re.sub(r"#if 0\s.*?#endif", "", source, flags=re.DOTALL)

    candidate_subs = {**_SHADER_SUBS, "FLOW_SOURCE_GENERATION_VALIDITY": 1}
    for shader_name in (*FLOW_SOURCE_PRODUCERS, "reactions/apply_bridge_flow_sources.comp"):
        control = strip_disabled_blocks(shader_source(shader_name, _SHADER_SUBS))
        candidate = shader_source(shader_name, candidate_subs)
        assert "flow_source_generation_validity_enabled" not in control
        assert "flow_source_generation_img" not in control
        assert "flow_source_generation_validity_enabled" in candidate

    fused_control = strip_disabled_blocks(
        shader_source("reactions/_self_fused_gas_output.comp", _SHADER_SUBS)
    )
    fused_candidate = shader_source(
        "reactions/_self_fused_gas_output.comp",
        candidate_subs,
    )
    assert "flow_source_generation_validity_enabled" not in fused_control
    assert "flow_source_generation_img" not in fused_control
    assert "flow_source_generation_validity_enabled" in fused_candidate


def test_flow_source_apply_filters_generation_before_payload_fetch() -> None:
    source = shader_source("reactions/apply_bridge_flow_sources.comp")
    generation_fetch = source.index("texelFetch(flow_source_generation_tex")
    payload_fetch = source.index("texelFetch(flow_source_tex")
    assert generation_fetch < payload_fetch
    assert "!= flow_source_generation" in source

    apply_source = inspect.getsource(_apply_flow_sources_to_bridge_velocity)
    assert 'program["flow_source_generation_validity_enabled"].value' in apply_source
    assert "resources.flow_source_generation_tex.use(location=2)" in apply_source


def test_flow_source_generation_binding_is_shared_by_all_writer_dispatches() -> None:
    helper_source = inspect.getsource(_bind_flow_source_generation_output)
    assert "if not pipeline._flow_source_generation_programs_enabled:" in helper_source
    assert '"flow_source_generation_validity_enabled"' in helper_source
    assert '"flow_source_generation"' in helper_source
    assert "flow_source_generation_tex.bind_to_image" in helper_source

    from oracle_game.sim import gpu_reactions_cell_pass
    from oracle_game.sim import gpu_reactions_pairings
    from oracle_game.sim import gpu_reactions_side_effects

    callsites = "\n".join(
        (
            inspect.getsource(gpu_reactions_cell_pass),
            inspect.getsource(gpu_reactions_pairings),
            inspect.getsource(gpu_reactions_side_effects),
        )
    )
    # One fused self producer, three gas pair/light producers, and four cell
    # action/side-effect producers all use the shared generation lifecycle.
    assert callsites.count("pipeline._bind_flow_source_generation_output(") == 8


def test_sparse_self_and_timed_paths_propagate_exact_flow_source_lifecycle() -> None:
    self_source = inspect.getsource(run_self_actions)
    assert self_source.count(
        "may_have_flow_sources=pipeline._compiled_actions_include_flow_sources(compiled)"
    ) == 2

    timed_source = inspect.getsource(run_timed_actions)
    assert "may_have_flow_sources=may_have_flow_sources" in timed_source
    assert "flow_source_layers=16" in timed_source

    timed_candidate_source = inspect.getsource(_run_timed_candidate_action_pass)
    assert "may_have_flow_sources: bool = False" in timed_candidate_source
    assert "flow_source_layers: int = 16" in timed_candidate_source
    assert "may_have_flow_sources=may_have_flow_sources" in timed_candidate_source
    assert "flow_source_layers=flow_source_layers" in timed_candidate_source


def test_flow_source_writer_to_sampler_paths_publish_texture_fetch_barriers() -> None:
    for function in (
        _run_cell_gas_action_delta_pass,
        _run_self_candidate_gas_side_effect_pass,
    ):
        source = inspect.getsource(function)
        assert "ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT" in source
        assert "ctx.TEXTURE_FETCH_BARRIER_BIT" in source
