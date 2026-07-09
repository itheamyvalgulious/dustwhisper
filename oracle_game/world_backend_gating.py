from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def use_cpu_oracle_backend(engine: "WorldEngine") -> None:
    engine.simulation_backend = "cpu"


def require_gpu_world_backend(engine: "WorldEngine") -> None:
    engine.simulation_backend = "gpu"


def prewarm_formal_connected_collapse(engine: "WorldEngine") -> bool:
    if engine.simulation_backend != "gpu":
        return False
    pipeline = engine.collapse_solver.gpu_pipeline
    if not pipeline.available(engine):
        return False
    pipeline.prewarm_formal_connected_resources(engine)
    return True


def _gpu_context_available(engine: "WorldEngine") -> bool:
    ctx = engine.bridge.ctx
    return bool(engine.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)


def _gpu_world_simulation_required(engine: "WorldEngine") -> bool:
    return engine.simulation_backend == "gpu"


def _gpu_realtime_budget_active(engine: "WorldEngine") -> bool:
    if not (engine.gpu_realtime_budget_enabled and engine.simulation_backend == "gpu"):
        return False
    active_tile_count = _gpu_active_tile_count(engine)
    if active_tile_count <= 0:
        return False
    estimated_active_cells = active_tile_count * int(engine.active.tile_size) * int(engine.active.tile_size)
    return estimated_active_cells >= int(engine.gpu_realtime_budget_cell_threshold)


def _gpu_active_tile_count(engine: "WorldEngine") -> int:
    if "active_tile_ttl" in engine.bridge.gpu_authoritative_resources:
        active_meta = engine.bridge.shadow_buffers.get("active_meta")
        if isinstance(active_meta, np.ndarray) and active_meta.size > 0:
            return int(active_meta[0]["active_tile_count"])
        return 0
    active_tile_ttl = np.asarray(engine.active.active_tile_ttl, dtype=np.int32)
    if active_tile_ttl.size <= 0:
        return 0
    return int(np.count_nonzero(active_tile_ttl > 0))


def _skip_budgeted_gpu_stage(engine: "WorldEngine", stage: str) -> bool:
    return False


def _should_run_formal_collapse_this_frame(engine: "WorldEngine") -> bool:
    if engine.simulation_backend != "gpu":
        return True
    interval = max(1, int(getattr(engine, "formal_collapse_interval_frames", 1)))
    if interval <= 1:
        return True
    frame_id = max(1, int(getattr(engine, "frame_id", 1)))
    return (frame_id - 1) % interval == 0


def _gpu_pipeline_available(engine: "WorldEngine", pipeline: Any, name: str, *, require: bool | None = None) -> bool:
    if engine.simulation_backend == "cpu":
        return False
    available = bool(pipeline.available(engine))
    required = _gpu_world_simulation_required(engine) if require is None else bool(require)
    if required and not available:
        raise RuntimeError(f"GPU world simulation requires the {name} GPU pipeline; CPU fallback is disabled")
    return available


def _require_gpu_stage(engine: "WorldEngine", name: str) -> None:
    if _gpu_world_simulation_required(engine):
        raise RuntimeError(f"GPU world simulation requires GPU support for {name}; CPU fallback is disabled")


def _require_gpu_authoritative_resources(engine: "WorldEngine", stage: str, *resource_names: str) -> None:
    if not (engine.simulation_backend == "gpu" and engine._world_simulation_frame_active):
        return
    missing = [str(name) for name in resource_names if str(name) not in engine.bridge.gpu_authoritative_resources]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"GPU world simulation requires GPU-authoritative {stage} resources: {joined}; "
            "CPU fallback is disabled"
        )


def _require_cpu_oracle_backend(engine: "WorldEngine", name: str) -> None:
    if engine.simulation_backend != "cpu":
        raise RuntimeError(
            f"{name} CPU oracle path requires simulation_backend='cpu'; CPU fallback is disabled"
        )


def _invalidate_gpu_authoritative_resources(engine: "WorldEngine", *resource_names: str) -> None:
    if engine.simulation_backend == "gpu":
        engine.bridge.clear_gpu_authoritative(*resource_names)
        engine._bridge_inputs_prepared = False
        if not engine._world_simulation_frame_active:
            engine._gpu_cpu_dirty_resources.update(str(name) for name in resource_names)


def _invalidate_gpu_authoritative_cell_resources(engine: "WorldEngine") -> None:
    _invalidate_gpu_authoritative_resources(
        engine,
        "cell_core",
        "material",
        "island_id",
        "entity_id",
        "placeholder_displaced_material",
        "collapse_delay_pending",
        "liquid_flow_intent",
    )
