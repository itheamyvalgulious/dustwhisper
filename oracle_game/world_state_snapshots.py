from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.gpu import unpack_cell_core
from oracle_game.types import EntityPlaceholder, EntityState, MaterialOpticsDef, ObservationTarget

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _material_optics_snapshot_map(engine: "WorldEngine") -> dict[tuple[str, str], MaterialOpticsDef]:
    payload = engine._stable_shadow_payload("optics", engine._material_optics_table_snapshot_payload)
    snapshot: dict[tuple[str, str], MaterialOpticsDef] = {}
    for item in payload:
        entry = engine._coerce_material_optics_def(item)
        snapshot[(entry.material_name, entry.light_type)] = entry
    return snapshot


def simulation_backend_report(engine: "WorldEngine") -> dict[str, Any]:
    ctx = engine.bridge.ctx
    gpu_available = bool(engine.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)
    ctx_info = getattr(ctx, "info", {}) if ctx is not None else {}
    backends = {
        "collapse": str(engine.collapse_solver.last_backend),
        "gas": str(engine.gas_solver.last_backend),
        "heat": str(engine.heat_solver.last_backend),
        "reactions": str(engine.reaction_solver.last_runtime_backend),
        "motion": str(engine.motion_solver.last_backend),
        "liquid": str(engine.liquid_solver.last_backend),
        "placeholder": str(engine.placeholder_pipeline.last_backend),
        "page_stripe": str(engine.page_stripe_pipeline.last_backend),
        "world_commands": str(engine.grid_command_pipeline.last_backend),
        "optics": str(engine.optics_solver.last_backend),
    }
    non_gpu = {name: backend for name, backend in backends.items() if backend not in {"gpu", "idle"}}
    return {
        "simulation_backend": engine.simulation_backend,
        "gpu_available": gpu_available,
        "renderer": str(ctx_info.get("GL_RENDERER", "")),
        "vendor": str(ctx_info.get("GL_VENDOR", "")),
        "opengl_version": str(ctx_info.get("GL_VERSION", "")),
        "gpu_realtime_budget": {
            "enabled": bool(engine.gpu_realtime_budget_enabled),
            "active": bool(engine._gpu_realtime_budget_active()),
            "cell_threshold": int(engine.gpu_realtime_budget_cell_threshold),
            "skipped_stages": list(engine.last_skipped_gpu_stages),
        },
        "backends": backends,
        "non_gpu_backends": non_gpu,
        "strict_gpu_ready": gpu_available and not non_gpu,
    }


def _preview_bridge_placeholder_dirty_rects(
    engine: "WorldEngine",
    current_entity_placeholders: dict[int, set[tuple[int, int]]],
    placeholders: list[EntityPlaceholder],
) -> list[dict[str, Any]]:
    current_cells = {
        cell: entity_id
        for entity_id, cells in current_entity_placeholders.items()
        for cell in cells
    }
    next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
    for placeholder in placeholders:
        for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
            for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                if not engine.in_bounds(x, y):
                    continue
                next_cells[(x, y)] = placeholder

    changed_cells: set[tuple[int, int]] = set()
    for cell, entity_id in current_cells.items():
        next_placeholder = next_cells.get(cell)
        if next_placeholder is None or next_placeholder.entity_id != entity_id:
            changed_cells.add(cell)
    for cell, placeholder in next_cells.items():
        x, y = cell
        material_id = int(engine.material_id[y, x])
        entity_id = int(engine.entity_id[y, x])
        has_matching_placeholder_cell = (
            material_id > 0
            and engine._shadow_material_is_placeholder(material_id)
            and entity_id == int(placeholder.entity_id)
        )
        if current_cells.get(cell) != placeholder.entity_id or not has_matching_placeholder_cell:
            changed_cells.add(cell)

    payload: list[dict[str, Any]] = []
    for x, y in sorted(changed_cells):
        world_rect = engine._buffer_bbox_to_world_bbox((int(x), int(y), int(x) + 1, int(y) + 1))
        payload.append(
            {
                "buffer_rect": [int(x), int(y), int(x) + 1, int(y) + 1],
                "world_rect": [int(world_rect[0]), int(world_rect[1]), int(world_rect[2]), int(world_rect[3])],
            }
        )
    return payload


def _bridge_shadow_buffer_coord_space(engine: "WorldEngine", array: np.ndarray) -> str | None:
    if array.ndim < 2:
        return None
    trailing_shape = tuple(int(value) for value in array.shape[-2:])
    if trailing_shape == (int(engine.height), int(engine.width)):
        return "world"
    if trailing_shape == (int(engine.gas_height), int(engine.gas_width)):
        return "gas"
    return None


def _current_cell_state_snapshot(engine: "WorldEngine", *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
    if (
        engine.simulation_backend == "gpu"
        and "cell_core" in engine.bridge.gpu_authoritative_resources
        and engine.bridge.enabled
        and engine.bridge.ctx is not None
        and "cell_core" in engine.bridge.buffers
    ):
        if not allow_gpu_sync_readback:
            return {
                "material_id": engine.material_id,
                "phase": engine.phase,
                "integrity": engine.integrity,
            }
        try:
            core = np.frombuffer(
                engine.bridge.buffers["cell_core"].read(size=engine.width * engine.height * 5 * np.dtype(np.uint32).itemsize),
                dtype=np.uint32,
            ).reshape((engine.height, engine.width, 5))
            unpacked = unpack_cell_core(core)
            return {
                "material_id": unpacked["material_id"].astype(np.int32, copy=False),
                "phase": unpacked["phase"].astype(np.uint8, copy=False),
                "integrity": unpacked["integrity"].astype(np.float32, copy=False),
            }
        except Exception as exc:
            raise RuntimeError(
                "GPU-authoritative cell state is not directly readable from this thread; "
                "use async readback for CPU-visible world snapshots"
            ) from exc
    return {
        "material_id": engine.material_id,
        "phase": engine.phase,
        "integrity": engine.integrity,
    }


def _current_entity_runtime_snapshot(engine: "WorldEngine", *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
    if (
        engine.simulation_backend == "gpu"
        and engine.bridge.enabled
        and engine.bridge.ctx is not None
        and "entity_id" in engine.bridge.gpu_authoritative_resources
        and "placeholder_displaced_material" in engine.bridge.gpu_authoritative_resources
        and "entity_id" in engine.bridge.buffers
        and "placeholder_displaced_material" in engine.bridge.buffers
    ):
        if not allow_gpu_sync_readback:
            return {
                "entity_id": engine.entity_id,
                "placeholder_displaced_material": engine.placeholder_displaced_material,
            }
        try:
            return {
                "entity_id": np.frombuffer(
                    engine.bridge.buffers["entity_id"].read(size=engine.entity_id.nbytes),
                    dtype=np.int32,
                ).reshape(engine.entity_id.shape),
                "placeholder_displaced_material": np.frombuffer(
                    engine.bridge.buffers["placeholder_displaced_material"].read(size=engine.placeholder_displaced_material.nbytes),
                    dtype=np.int32,
                ).reshape(engine.placeholder_displaced_material.shape),
            }
        except Exception as exc:
            raise RuntimeError(
                "GPU-authoritative entity runtime state is not directly readable from this thread; "
                "use async readback for CPU-visible world snapshots"
            ) from exc
    return {
        "entity_id": engine.entity_id,
        "placeholder_displaced_material": engine.placeholder_displaced_material,
    }


def _entity_placeholder_state_gpu_authoritative(engine: "WorldEngine") -> bool:
    if engine.simulation_backend != "gpu":
        return False
    authoritative = engine.bridge.gpu_authoritative_resources
    return bool(
        "cell_core" in authoritative
        or "entity_id" in authoritative
        or "placeholder_displaced_material" in authoritative
    )


def _runtime_entities_to_immediate_observation_targets(
    engine: "WorldEngine",
    entities: list[EntityState],
) -> list[ObservationTarget]:
    targets: list[ObservationTarget] = []
    for entity in entities:
        if not entity.observe_channels:
            continue
        if entity.world_x is not None and entity.world_y is not None:
            world_x = int(entity.world_x)
            world_y = int(entity.world_y)
        else:
            world_x, world_y = engine._buffer_to_world_position((int(entity.x), int(entity.y)))
        entity_width = max(1, int(entity.width))
        entity_height = max(1, int(entity.height))
        center_x = int((world_x + world_x + entity_width - 1) // 2)
        center_y = int((world_y + world_y + entity_height - 1) // 2)
        width = int(entity.observe_width) if entity.observe_width is not None else entity_width + int(entity.observe_pad_cells) * 2
        height = int(entity.observe_height) if entity.observe_height is not None else entity_height + int(entity.observe_pad_cells) * 2
        targets.append(
            ObservationTarget(
                observer_id=int(entity.entity_id),
                center_x=int(center_x),
                center_y=int(center_y),
                width=max(1, int(width)),
                height=max(1, int(height)),
                channels=entity.observe_channels,
                pad_cells=int(entity.observe_pad_cells),
                label=entity.observe_label,
            )
        )
    return targets
