"""Read windows, bridge snapshots, readback state, and cell-core layout keys."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS


if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _capabilities_views_section(engine: "WorldEngine", ctx) -> dict[str, Any]:
    """Build the views portion of the engine capabilities schema."""

    emitter_fields = ctx.emitter_fields
    cell_window_fields = ctx.cell_window_fields
    temperature_window_fields = ctx.temperature_window_fields
    gas_window_fields = ctx.gas_window_fields
    pressure_window_fields = ctx.pressure_window_fields
    velocity_window_fields = ctx.velocity_window_fields
    optics_window_fields = ctx.optics_window_fields
    visible_illumination_fields = ctx.visible_illumination_fields
    debug_frame_fields = ctx.debug_frame_fields
    bridge_runtime_fields = ctx.bridge_runtime_fields
    bridge_resource_catalog_fields = ctx.bridge_resource_catalog_fields
    bridge_typed_table_fields = ctx.bridge_typed_table_fields
    bridge_typed_table_slice_fields = ctx.bridge_typed_table_slice_fields
    bridge_shadow_buffer_fields = ctx.bridge_shadow_buffer_fields
    bridge_shadow_buffer_slice_fields = ctx.bridge_shadow_buffer_slice_fields
    bridge_shadow_buffer_window_fields = ctx.bridge_shadow_buffer_window_fields
    bridge_shadow_buffer_spatial_window_fields = ctx.bridge_shadow_buffer_spatial_window_fields
    bridge_upload_snapshot_fields = ctx.bridge_upload_snapshot_fields
    bridge_readback_stage_fields = ctx.bridge_readback_stage_fields
    bridge_index_stage_fields = ctx.bridge_index_stage_fields
    bridge_frame_snapshot_fields = ctx.bridge_frame_snapshot_fields
    readback_state_fields = ctx.readback_state_fields
    readback_result_fields = ctx.readback_result_fields
    readback_poll_fields = ctx.readback_poll_fields
    readback_plan_fields = ctx.readback_plan_fields
    observation_plan_fields = ctx.observation_plan_fields
    readback_channel_payload_types = ctx.readback_channel_payload_types
    readback_payload_fields = ctx.readback_payload_fields
    readback_cell_payload_fields = ctx.readback_cell_payload_fields
    readback_scalar_window_fields = ctx.readback_scalar_window_fields
    readback_vector_window_fields = ctx.readback_vector_window_fields
    readback_gas_payload_fields = ctx.readback_gas_payload_fields
    readback_optics_payload_fields = ctx.readback_optics_payload_fields
    emitter_runtime_fields = ctx.emitter_runtime_fields
    cell_core_layout_fields = ctx.cell_core_layout_fields
    cell_core_unpack_fields = ctx.cell_core_unpack_fields

    return {
        "persistent_emitters": {
            "fields": emitter_fields,
            "replace_semantics": "replace_all",
        },
        "emitters": {
            "fields": emitter_runtime_fields,
            "field_types": {
                "persistent_emitters": {"type": "emitter[]"},
                "queued_emitters": {"type": "emitter[]"},
            },
        },
        "force_sources_response": {
            "fields": ["force_sources"],
            "field_types": {
                "force_sources": {"type": "force_source[]"},
            },
        },
        "cell_window": {
            "fields": cell_window_fields,
            "origin_type": "cell_xy",
            "size_type": "cell_wh",
            "temperature_alias": "temperature",
        },
        "temperature_window": {
            "fields": temperature_window_fields,
            "grid": "cell",
        },
        "gas_window": {
            "fields": gas_window_fields,
            "grid": "gas",
            "species_key": "species",
            "concentration_key": "concentration",
        },
        "pressure_window": {
            "fields": pressure_window_fields,
            "grid": "gas",
        },
        "velocity_window": {
            "fields": velocity_window_fields,
            "grid": "gas",
            "vector_components": 2,
        },
        "optics_window": {
            "fields": optics_window_fields,
            "origin_type": "cell_xy",
            "size_type": "cell_wh",
            "gas_origin_type": "gas_xy",
            "gas_size_type": "gas_wh",
            "dose_key_type": "light_name",
        },
        "visible_illumination": {
            "fields": visible_illumination_fields,
            "grid": "cell",
            "color_components": 3,
        },
        "debug_frame": {
            "fields": debug_frame_fields,
            "origin_type": "cell_xy",
            "size_type": "cell_wh",
            "frame_type": "rgb_grid",
            "color_components": 3,
        },
        "bridge_runtime": {
            "fields": bridge_runtime_fields,
            "field_types": {
                "frame_id": {"type": "int"},
                "bridge": {"type": "json"},
                "pending_readbacks": {"type": "int"},
                "inflight_readbacks": {"type": "int"},
                "ready_readbacks": {"type": "int"},
                "pending_commands": {"type": "int"},
            },
        },
        "bridge_resource_catalog": {
            "fields": bridge_resource_catalog_fields,
            "field_types": {
                "typed_tables": {"type": "json"},
                "shadow_buffers": {"type": "json"},
                "snapshots": {"type": "json"},
            },
        },
        "bridge_typed_table_snapshot": {
            "fields": bridge_typed_table_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "rows": {"type": "json"},
            },
        },
        "bridge_typed_table_slice_snapshot": {
            "fields": bridge_typed_table_slice_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "offset": {"type": "int"},
                "limit": {"type": "int"},
                "returned_count": {"type": "int"},
                "slice_shape": {"type": "int[]"},
                "rows": {"type": "json"},
            },
        },
        "bridge_shadow_buffer_snapshot": {
            "fields": bridge_shadow_buffer_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "rows": {"type": "json"},
                "values": {"type": "json"},
                "utf8": {"type": "string", "optional": True},
            },
        },
        "bridge_shadow_buffer_slice_snapshot": {
            "fields": bridge_shadow_buffer_slice_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "offset": {"type": "int"},
                "limit": {"type": "int"},
                "returned_count": {"type": "int"},
                "slice_shape": {"type": "int[]"},
                "rows": {"type": "json"},
                "values": {"type": "json"},
                "utf8": {"type": "string", "optional": True},
            },
        },
        "bridge_shadow_buffer_window_snapshot": {
            "fields": bridge_shadow_buffer_window_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "window_origin": {"type": "int[]"},
                "requested_size": {"type": "int[]"},
                "window_size": {"type": "int[]"},
                "window_axes": {"type": "int[]"},
                "returned_shape": {"type": "int[]"},
                "rows": {"type": "json"},
                "values": {"type": "json"},
                "utf8": {"type": "string", "optional": True},
            },
        },
        "bridge_shadow_buffer_world_window_snapshot": {
            "fields": bridge_shadow_buffer_spatial_window_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "coord_space": {"type": "string"},
                "window_origin": {"type": "int[]"},
                "requested_size": {"type": "int[]"},
                "window_size": {"type": "int[]"},
                "window_axes": {"type": "int[]"},
                "returned_shape": {"type": "int[]"},
                "rows": {"type": "json"},
                "values": {"type": "json"},
                "utf8": {"type": "string", "optional": True},
            },
        },
        "bridge_shadow_buffer_gas_window_snapshot": {
            "fields": bridge_shadow_buffer_spatial_window_fields,
            "field_types": {
                "name": {"type": "string"},
                "shape": {"type": "int[]"},
                "dtype": {"type": "string"},
                "structured": {"type": "bool"},
                "field_names": {"type": "string[]"},
                "row_count": {"type": "int"},
                "coord_space": {"type": "string"},
                "window_origin": {"type": "int[]"},
                "requested_size": {"type": "int[]"},
                "window_size": {"type": "int[]"},
                "window_axes": {"type": "int[]"},
                "returned_shape": {"type": "int[]"},
                "rows": {"type": "json"},
                "values": {"type": "json"},
                "utf8": {"type": "string", "optional": True},
            },
        },
        "bridge_upload_snapshot": {
            "fields": bridge_upload_snapshot_fields,
            "field_types": {
                "frame_meta": {"type": "json"},
                "world_commands": {"type": "world_command[]"},
                "readback_requests": {"type": "json"},
                "page_stripes": {"type": "json"},
            },
        },
        "bridge_frame_snapshot": {
            "fields": bridge_frame_snapshot_fields,
            "readback_stage_type": "bridge_readback_stage",
            "index_stage_type": "bridge_index_stage",
            "field_types": {
                "prepared": {"type": "bool"},
                "commands": {"type": "world_command[]"},
                "command_stages": {"type": "bridge_index_stage[]"},
                "readback_requests": {"type": "readback_request[]"},
                "readback_request_stages": {"type": "bridge_readback_stage[]"},
                "placeholders": {"type": "entity_placeholder_runtime[]"},
                "placeholder_stages": {"type": "bridge_index_stage[]"},
                "placeholder_dirty_rects": {"type": "json"},
                "paging_updates": {"type": "page_stripe_update[]"},
                "paging_update_stages": {"type": "bridge_index_stage[]"},
                "page_stripes": {"type": "json"},
                "page_stripe_stages": {"type": "bridge_index_stage[]"},
            },
        },
        "bridge_readback_stage": {
            "fields": bridge_readback_stage_fields,
        },
        "bridge_index_stage": {
            "fields": bridge_index_stage_fields,
        },
        "readback_state": {
            "fields": readback_state_fields,
            "request_type": "readback_request",
            "queued_command_type": "world_command",
        },
        "readback_result": {
            "fields": readback_result_fields,
            "request_type": "readback_request",
            "payload_type": "json",
            "payload_schema_type": "readback_payload",
        },
        "readback_poll": {
            "fields": readback_poll_fields,
            "result_type": "readback_result",
            "result_optional_when_not_ready": True,
        },
        "readback_ready": {
            "fields": ["ready", "results"],
            "field_types": {
                "ready": {"type": "int"},
                "results": {"type": "readback_result[]"},
            },
        },
        "readback_poll_all": {
            "fields": ["results"],
            "field_types": {
                "results": {"type": "readback_result[]"},
            },
        },
        "readback_plan": {
            "fields": readback_plan_fields,
            "request_type": "readback_request",
            "layout_type": "json",
            "payload_type": "json",
            "field_types": {
                "request": {"type": "readback_request"},
                "layout": {"type": "json"},
                "nbytes": {"type": "int"},
                "gpu_source_count": {"type": "int"},
                "cpu_chunk_count": {"type": "int"},
                "payload": {"type": "json"},
            },
        },
        "observation_plan": {
            "fields": observation_plan_fields,
            "target_type": "observation_target",
            "request_type": "readback_request",
            "layout_type": "json",
            "payload_type": "json",
            "field_types": {
                "target": {"type": "observation_target"},
                "request": {"type": "readback_request"},
                "layout": {"type": "json"},
                "nbytes": {"type": "int"},
                "gpu_source_count": {"type": "int"},
                "cpu_chunk_count": {"type": "int"},
                "payload": {"type": "json"},
            },
        },
        "readback_payload": {
            "fields": readback_payload_fields,
            "channel_types": dict(readback_channel_payload_types),
            "optional_fields": list(READBACK_ALLOWED_CHANNELS),
        },
        "readback_cell_payload": {
            "fields": readback_cell_payload_fields,
            "origin_type": "cell_xy",
            "size_type": "cell_wh",
            "packed_core_words": True,
            "core_words_layout_type": "cell_core_layout",
        },
        "readback_scalar_window": {
            "fields": readback_scalar_window_fields,
            "origin_type": "gas_xy",
            "size_type": "gas_wh",
            "grid_key": "grid",
            "values_type": "scalar_grid",
        },
        "readback_vector_window": {
            "fields": readback_vector_window_fields,
            "origin_type": "gas_xy",
            "size_type": "gas_wh",
            "grid_key": "grid",
            "values_type": "vector_grid",
            "vector_components": 2,
        },
        "readback_gas_payload": {
            "fields": readback_gas_payload_fields,
            "origin_type": "gas_xy",
            "size_type": "gas_wh",
            "species_key_type": "gas_name",
        },
        "readback_optics_payload": {
            "fields": readback_optics_payload_fields,
            "origin_type": "cell_xy",
            "size_type": "cell_wh",
            "gas_origin_type": "gas_xy",
            "gas_size_type": "gas_wh",
            "dose_key_type": "light_name",
        },
        "cell_core_layout": {
            "fields": cell_core_layout_fields,
            "word_count": 5,
            "word_bits": 32,
            "packed_words": {
                "word0": "material_id:u16 | phase:u8 | cell_flags:u8",
                "word1": "velocity_pack = packHalf2x16(velocity_xy)",
                "word2": "cell_temperature:f32",
                "word3": "timer_pack:u32",
                "word4": "integrity:u16 | reserved0:u16",
            },
            "unpacked_fields": cell_core_unpack_fields,
            "unpack_schema": {
                "material_id": {"source": "word0", "dtype": "u16"},
                "phase": {"source": "word0", "dtype": "u8", "bit_range": [16, 23]},
                "cell_flags": {"source": "word0", "dtype": "u8", "bit_range": [24, 31]},
                "velocity": {
                    "source": "word1",
                    "encoding": "packHalf2x16",
                    "component_count": 2,
                    "dtype": "f16->f32",
                },
                "cell_temperature": {"source": "word2", "dtype": "f32"},
                "timer_pack": {
                    "source": "word3",
                    "component_count": 4,
                    "component_dtype": "u8",
                },
                "integrity": {"source": "word4", "dtype": "u16->f32", "bit_range": [0, 15]},
            },
        },
    }
