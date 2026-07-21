from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine

from oracle_game.sim.gpu_reactions import LOCAL_SIZE


def _run_timed_self_combined_action_pass(
    pipeline,
    world: "WorldEngine",
    resources,
    *,
    self_rule_count: int,
    modify_gas_layer_mask: int,
    modifies_gas: bool,
    use_expanded_active_tile_mask: bool = False,
) -> bool:
    """Run timed core then self core in one invocation per cell.

    The candidate is deliberately limited to formal GPU segments. Timed and
    self deferred vectors remain separate so the post-dispatch gas scatter
    preserves each stage's eight-action capacity and ordering.
    """
    ctx = world.bridge.ctx
    assert ctx is not None
    segment_key = pipeline._formal_segment_batch_key
    if segment_key is None:
        raise RuntimeError("timed/self combined dispatch requires an active formal segment")
    if modifies_gas:
        pipeline._clear_formal_segment_gas_delta(world, resources, segment_key)

    program = pipeline.programs["timed_self_apply_combined"]
    pipeline._set_uniform_if_present(program, "cell_grid_size", (world.width, world.height))
    pipeline._set_uniform_if_present(program, "rule_count", 0)
    pipeline._set_uniform_if_present(program, "gas_cell_size", int(world.gas_cell_size))
    pipeline._set_uniform_if_present(program, "gas_count", int(world.gas_concentration.shape[0]))
    pipeline._set_uniform_if_present(program, "random_target_count", int(pipeline.random_target_count))
    pipeline._set_uniform_if_present(program, "self_rule_count", int(self_rule_count))
    pipeline._set_uniform_if_present(
        program,
        "use_self_rule_material_spans",
        bool(pipeline._self_rule_material_spans_enabled),
    )
    pipeline._set_uniform_if_present(program, "gas_grid_size", (world.gas_width, world.gas_height))
    pipeline._set_uniform_if_present(program, "direct_gas_delta_enabled", False)
    pipeline._set_uniform_if_present(program, "direct_modify_gas_layer_mask", int(modify_gas_layer_mask))
    pipeline._set_uniform_if_present(program, "write_deferred_hi_outputs", True)
    pipeline._set_uniform_if_present(program, "collect_self_emit_targets", False)
    pipeline._set_uniform_if_present(program, "collect_self_gas_candidates", False)
    pipeline._set_uniform_if_present(
        program,
        "use_expanded_active_tile_mask",
        bool(use_expanded_active_tile_mask),
    )

    cell_state_in, _phase_in, temp_in, integrity_in, velocity_in, timer_in = (
        pipeline._current_cell_textures(resources)
    )
    cell_state_in.use(location=0)
    temp_in.use(location=2)
    integrity_in.use(location=3)
    resources.gas_ping.use(location=4)
    resources.cell_dose_tex.use(location=5)
    timer_in.use(location=6)
    resources.active_cell_tex.use(location=7)
    resources.expanded_active_tile_tex.use(location=1)
    velocity_in.use(location=8)
    resources.material_params.bind_to_storage_buffer(binding=0)
    resources.action_i.bind_to_storage_buffer(binding=1)
    resources.action_f.bind_to_storage_buffer(binding=2)
    resources.material_tags.bind_to_storage_buffer(binding=6)
    resources.gas_tags.bind_to_storage_buffer(binding=7)
    resources.material_slots_lo.bind_to_storage_buffer(binding=8)
    resources.material_slots_hi.bind_to_storage_buffer(binding=9)
    resources.action_meta.bind_to_storage_buffer(binding=10)
    resources.random_targets.bind_to_storage_buffer(binding=11)
    resources.self_rule_i.bind_to_storage_buffer(binding=12)
    resources.self_rule_f.bind_to_storage_buffer(binding=13)
    resources.light_emitter_buffer.bind_to_storage_buffer(binding=14)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)

    cell_state_out, _phase_out, temp_out, integrity_out, _velocity_out, timer_out = (
        pipeline._next_cell_textures(resources)
    )
    cell_state_out.bind_to_image(0, read=False, write=True)
    temp_out.bind_to_image(2, read=False, write=True)
    integrity_out.bind_to_image(3, read=False, write=True)
    timer_out.bind_to_image(4, read=False, write=True)
    resources.local_deferred_packed_out.bind_to_image(5, read=False, write=True)
    resources.local_deferred_lo_out.bind_to_image(6, read=False, write=True)
    resources.local_deferred_hi_out.bind_to_image(7, read=False, write=True)

    group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
    group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
    with pipeline._profile_pass(world, "timed_self_combined_shader"):
        program.run(group_x, group_y, 1)
        pipeline._sync_compute_writes(ctx)

    if not modifies_gas:
        return False
    scatter = pipeline.programs["scatter_timed_self_gas_action_delta"]
    scatter["cell_grid_size"].value = (world.width, world.height)
    scatter["gas_grid_size"].value = (world.gas_width, world.gas_height)
    scatter["gas_cell_size"].value = int(world.gas_cell_size)
    scatter["gas_count"].value = int(world.gas_concentration.shape[0])
    scatter["modify_gas_layer_mask"].value = int(modify_gas_layer_mask)
    pipeline._set_uniform_if_present(
        scatter,
        "use_expanded_active_tile_mask",
        bool(use_expanded_active_tile_mask),
    )
    resources.local_deferred_lo_out.use(location=0)
    resources.local_deferred_hi_out.use(location=1)
    resources.local_deferred_packed_out.use(location=2)
    resources.active_cell_tex.use(location=3)
    resources.expanded_active_tile_tex.use(location=4)
    resources.action_i.bind_to_storage_buffer(binding=0)
    resources.action_f.bind_to_storage_buffer(binding=1)
    resources.gas_delta_buffer.bind_to_storage_buffer(binding=2)
    resources.light_emitter_count.bind_to_storage_buffer(binding=15)
    with pipeline._profile_pass(world, "timed_self_combined_gas_scatter"):
        scatter.run(group_x, group_y, 1)
        ctx.memory_barrier(ctx.SHADER_STORAGE_BARRIER_BIT | ctx.TEXTURE_FETCH_BARRIER_BIT)
    return False
