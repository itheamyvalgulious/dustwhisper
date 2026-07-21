from __future__ import annotations

import inspect

from oracle_game.sim.gpu_reactions import GPUReactionPipeline, _SHADER_SUBS
from oracle_game.sim.gpu_reactions_bridge import _apply_flow_sources_to_bridge_velocity
from oracle_game.sim.gpu_reactions_resources import _ensure_resources
from oracle_game.sim.gpu_reactions_transient import _advance_flow_source_generation
from oracle_game.sim.shader_loader import shader_source


FLOW_SOURCE_WRITERS = (
    "reactions/scatter_cell_gas_action_delta.comp",
    "reactions/scatter_cell_gas_action_delta_candidates.comp",
    "reactions/scatter_self_gas_action_delta_candidates.comp",
    "reactions/cell_gas_side_effects.comp",
    "reactions/gas_gas.comp",
    "reactions/gas_light.comp",
    "reactions/_self_fused_gas_output.comp",
    "reactions/clear_transient_flow_source_generations.comp",
)


def test_flow_source_u8_generation_token_is_default_on_and_compile_isolated() -> None:
    pipeline = GPUReactionPipeline()
    assert pipeline._flow_source_generation_u8_token_enabled is True
    assert pipeline._flow_source_generation_u8_programs_enabled is False
    assert _SHADER_SUBS["FLOW_SOURCE_GENERATION_IMAGE_FORMAT"] == "r32ui"

    control_subs = {
        **_SHADER_SUBS,
        "FLOW_SOURCE_GENERATION_VALIDITY": 1,
    }
    candidate_subs = {
        **control_subs,
        "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": "r8ui",
    }
    for shader_name in FLOW_SOURCE_WRITERS:
        control = shader_source(shader_name, control_subs)
        candidate = shader_source(shader_name, candidate_subs)
        assert "layout(r32ui" in control
        assert "layout(r8ui" not in control
        assert "layout(r8ui" in candidate
        assert "layout(r32ui" not in candidate or "emit_timer_img" in candidate


def test_flow_source_u8_resource_has_separate_u1_and_u4_paths() -> None:
    source = inspect.getsource(_ensure_resources)
    assert '"u1" if pipeline._flow_source_generation_u8_programs_enabled else "u4"' in source
    assert "np.uint8" in source
    assert "np.uint32" in source
    assert "dtype=flow_generation_dtype" in source
    assert "dtype=flow_generation_numpy_dtype" in source


def test_flow_source_u8_wrap_clears_before_token_one_is_reused() -> None:
    source = inspect.getsource(_advance_flow_source_generation)
    assert "np.iinfo(np.uint8).max" in source
    assert "np.iinfo(np.uint32).max" in source
    clear = source.index('pipeline.programs["clear_transient_flow_source_generations"]')
    reset = source.index("generation = 0")
    advance = source.index("resources.flow_source_generation = generation + 1")
    assert clear < reset < advance


def test_flow_source_u8_keeps_canonical_reader_layer_loop() -> None:
    shader = shader_source(
        "reactions/apply_bridge_flow_sources.comp",
        {
            **_SHADER_SUBS,
            "FLOW_SOURCE_GENERATION_VALIDITY": 1,
            "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": "r8ui",
        },
    )
    assert "uniform usampler2DArray flow_source_generation_tex;" in shader
    assert "for (int layer = 0; layer < layer_count; ++layer)" in shader
    assert "texelFetch(flow_source_generation_tex" in shader
    assert "texelFetch(flow_source_tex" in shader

    host = inspect.getsource(_apply_flow_sources_to_bridge_velocity)
    assert 'program["flow_source_generation"].value' in host
    assert "resources.flow_source_generation_tex.use(location=2)" in host


def test_flow_source_u8_does_not_add_worklist_or_atomic_paths() -> None:
    sources = "\n".join(
        shader_source(
            shader_name,
            {
                **_SHADER_SUBS,
                "FLOW_SOURCE_GENERATION_VALIDITY": 1,
                "FLOW_SOURCE_GENERATION_IMAGE_FORMAT": "r8ui",
            },
        )
        for shader_name in FLOW_SOURCE_WRITERS
    )
    assert "imageAtomic" not in sources
    assert "flow_source_cell_worklist" not in sources
