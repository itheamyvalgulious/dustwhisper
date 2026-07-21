#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import moderngl

from oracle_game.types import ForceSource, Phase
from oracle_game.world import WorldEngine


ScenarioSetup = Callable[[WorldEngine], None]


def create_context() -> tuple[object, str]:
    errors: list[str] = []
    for label, kwargs in (
        ("egl", {"require": 430, "backend": "egl"}),
        ("default", {"require": 430}),
    ):
        try:
            return moderngl.create_standalone_context(**kwargs), label
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{label}: {exc}")
    raise RuntimeError("unable to create ModernGL context: " + " | ".join(errors))


def fill_rect(engine: WorldEngine, x0: int, y0: int, x1: int, y1: int, material: str, phase: Phase | None = None) -> None:
    for y in range(max(0, y0), min(engine.height, y1)):
        for x in range(max(0, x0), min(engine.width, x1)):
            engine.set_cell(x, y, material, phase=phase, mark_dirty=False)
    engine.active.mark_rect(x0, y0, x1, y1)


def activate_all(engine: WorldEngine) -> None:
    engine.active.mark_rect(0, 0, engine.width, engine.height)


def setup_random_materials(engine: WorldEngine) -> None:
    rng = np.random.default_rng(1729)
    material_ids = np.asarray(sorted(material_id for material_id in engine.rulebook.materials_by_id if material_id > 0), dtype=np.int32)
    if material_ids.size == 0:
        activate_all(engine)
        return
    phase_lookup = np.zeros((int(material_ids.max()) + 1,), dtype=np.uint8)
    for material_id, material in engine.rulebook.materials_by_id.items():
        if 0 <= int(material_id) < phase_lookup.size:
            phase_lookup[int(material_id)] = int(material.default_phase)
    engine.material_id[:, :] = rng.choice(material_ids, size=engine.material_id.shape, replace=True)
    engine.phase[:, :] = phase_lookup[engine.material_id]
    engine.cell_flags[:, :] = 0
    engine.velocity[:, :, :] = rng.normal(0.0, 0.15, size=engine.velocity.shape).astype(np.float32)
    engine.cell_temperature[:, :] = rng.normal(20.0, 12.0, size=engine.cell_temperature.shape).astype(np.float32)
    engine.timer_pack[:, :, :] = 0
    engine.integrity[:, :] = rng.uniform(10.0, 100.0, size=engine.integrity.shape).astype(np.float32)
    engine.island_id[:, :] = 0
    engine.entity_id[:, :] = 0
    engine.placeholder_displaced_material[:, :] = 0
    engine.flow_velocity[:, :, :] = rng.normal(0.0, 0.2, size=engine.flow_velocity.shape).astype(np.float32)
    engine.ambient_temperature[:, :] = rng.normal(20.0, 4.0, size=engine.ambient_temperature.shape).astype(np.float32)
    engine.gas_concentration[:, :, :] = rng.uniform(0.0, 0.05, size=engine.gas_concentration.shape).astype(np.float32)
    activate_all(engine)


def setup_empty_active(engine: WorldEngine) -> None:
    activate_all(engine)
    engine.force_sources.append(
        ForceSource(x=engine.width * 0.5, y=engine.height * 0.5, direction=(1.0, -0.25), radius=18.0, strength=1.2, lifetime=5.0)
    )


def setup_dense_liquid(engine: WorldEngine) -> None:
    fill_rect(engine, 8, engine.height // 2, engine.width - 8, engine.height - 8, "water_liquid", Phase.LIQUID)
    fill_rect(engine, 0, engine.height - 8, engine.width, engine.height, "raw_stone_solid", Phase.STATIC_SOLID)
    activate_all(engine)


def setup_dense_gas(engine: WorldEngine) -> None:
    water_gas = engine.rulebook.gas_id("water_gas")
    engine.gas_concentration[water_gas, engine.gas_height // 4 : engine.gas_height * 3 // 4, :] = 0.9
    engine.ambient_temperature[:, :] = 55.0
    engine.force_sources.append(
        ForceSource(x=engine.width * 0.35, y=engine.height * 0.45, direction=(1.0, 0.0), radius=24.0, strength=2.0, lifetime=5.0)
    )
    activate_all(engine)


def setup_optics_branches(engine: WorldEngine) -> None:
    fill_rect(engine, engine.width // 2, 8, engine.width // 2 + 3, engine.height - 8, "gold_solid", Phase.STATIC_SOLID)
    engine.emitters.append(
        {
            "light_type": "visible_light",
            "origin": (8, engine.height // 2),
            "direction": (1.0, 0.0),
            "spread": 0.25,
            "strength": 1.0,
            "range_cells": min(engine.width, 96),
        }
    )
    activate_all(engine)


def setup_mixed_reaction_motion(engine: WorldEngine) -> None:
    fill_rect(engine, 4, engine.height - 20, engine.width - 4, engine.height - 8, "raw_stone_solid", Phase.STATIC_SOLID)
    fill_rect(engine, 12, 12, engine.width // 2, 44, "sand_powder", Phase.POWDER)
    fill_rect(engine, engine.width // 2, 18, engine.width - 16, 42, "acid_liquid", Phase.LIQUID)
    fill_rect(engine, engine.width // 2 - 6, 18, engine.width // 2 - 3, 42, "raw_stone_solid", Phase.STATIC_SOLID)
    engine.emitters.append(
        {
            "light_type": "chaos_light",
            "origin": (engine.width // 2, 8),
            "direction": (0.0, 1.0),
            "spread": 0.2,
            "strength": 1.0,
            "range_cells": 80,
        }
    )
    engine.force_sources.append(
        ForceSource(x=engine.width * 0.25, y=engine.height * 0.25, direction=(0.6, 0.1), radius=20.0, strength=1.5, lifetime=5.0)
    )
    activate_all(engine)


SCENARIOS: dict[str, ScenarioSetup] = {
    "empty_active": setup_empty_active,
    "dense_liquid": setup_dense_liquid,
    "dense_gas": setup_dense_gas,
    "optics_branches": setup_optics_branches,
    "mixed_reaction_motion": setup_mixed_reaction_motion,
    "random_materials": setup_random_materials,
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def summarize_pass_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        for entry in profile.get("passes", []):
            name = str(entry.get("name", ""))
            if not name:
                continue
            aggregate = summary.setdefault(name, {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
            aggregate["count"] += 1
            aggregate["cpu_ms"] += float(entry.get("cpu_ms") or 0.0)
            gpu_ms = entry.get("gpu_ms")
            if gpu_ms is not None:
                aggregate["gpu_ms"] = float(aggregate["gpu_ms"] or 0.0) + float(gpu_ms)
        for child_name in ("collapse", "gas", "heat", "motion", "liquid", "optics", "reactions"):
            child_profile = profile.get(child_name)
            if not isinstance(child_profile, dict):
                continue
            for entry in child_profile.get("passes", []):
                name = f"{child_name}.{entry.get('name', '')}"
                if name == f"{child_name}.":
                    continue
                aggregate = summary.setdefault(name, {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
                aggregate["count"] += 1
                aggregate["cpu_ms"] += float(entry.get("cpu_ms") or 0.0)
                gpu_ms = entry.get("gpu_ms")
                if gpu_ms is not None:
                    aggregate["gpu_ms"] = float(aggregate["gpu_ms"] or 0.0) + float(gpu_ms)
    return {
        name: {
            "count": values["count"],
            "total_cpu_ms": values["cpu_ms"],
            "avg_cpu_ms": values["cpu_ms"] / values["count"] if values["count"] else 0.0,
            "total_gpu_ms": values["gpu_ms"],
            "avg_gpu_ms": (values["gpu_ms"] / values["count"]) if values["gpu_ms"] is not None and values["count"] else None,
        }
        for name, values in sorted(summary.items())
    }


def run_scenario(
    name: str,
    setup: ScenarioSetup,
    *,
    ctx: object,
    width: int,
    height: int,
    warmup: int,
    frames: int,
    dt: float,
    readback: bool,
    profile_passes: bool,
    profile_passes_sync: bool = False,
    heat_terminal_phase_fusion: bool | None = None,
    heat_terminal_dirty_publish_fusion: bool | None = None,
    heat_terminal_workgroup16x8: bool | None = None,
) -> dict[str, object]:
    engine = WorldEngine(width=width, height=height, gpu_context=ctx)
    try:
        engine.profile_passes_enabled = bool(profile_passes)
        engine.profile_passes_sync = bool(profile_passes_sync)
        heat_solver = getattr(engine, "heat_solver", None)
        heat_pipeline = getattr(heat_solver, "gpu_pipeline", None)
        if heat_pipeline is not None:
            if heat_terminal_phase_fusion is not None:
                heat_pipeline._terminal_phase_fusion_enabled = bool(heat_terminal_phase_fusion)
            if heat_terminal_dirty_publish_fusion is not None:
                heat_pipeline._terminal_dirty_publish_fusion_enabled = bool(
                    heat_terminal_dirty_publish_fusion
                )
            if heat_terminal_workgroup16x8 is not None:
                heat_pipeline._terminal4x6_workgroup16x8_enabled = bool(
                    heat_terminal_workgroup16x8
                )
        active_heat_terminal_phase_fusion = bool(
            getattr(heat_pipeline, "_terminal_phase_fusion_enabled", False)
        )
        active_heat_terminal_dirty_publish_fusion = bool(
            getattr(heat_pipeline, "_terminal_dirty_publish_fusion_enabled", False)
        )
        active_heat_terminal_workgroup16x8 = bool(
            getattr(heat_pipeline, "_terminal4x6_workgroup16x8_enabled", False)
        )
        setup(engine)
        if engine.simulation_backend == "gpu":
            engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
            engine._gpu_cpu_dirty_resources.clear()
            prewarm_collapse = getattr(engine, "prewarm_formal_connected_collapse", None)
            if callable(prewarm_collapse):
                prewarm_collapse()
        for _ in range(warmup):
            if readback:
                engine.request_readback(width // 2, height // 2, 8, 8, ("cell", "gas", "optics"), label=f"{name}_warmup")
            engine.step(dt)
            ctx.finish()
            engine.poll_all_readbacks()

        frame_ms: list[float] = []
        frame_samples: list[dict[str, object]] = []
        pass_profiles: list[dict[str, Any]] = []
        readbacks_completed = 0
        for frame_index in range(frames):
            if readback:
                engine.request_readback(width // 2, height // 2, 8, 8, ("cell", "gas", "optics"), label=f"{name}_{frame_index}")
            start = time.perf_counter()
            engine.step(dt)
            ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            frame_ms.append(elapsed_ms)
            collapse_pipeline = getattr(getattr(engine, "collapse_solver", None), "gpu_pipeline", None)
            epoch = getattr(collapse_pipeline, "_formal_dirty_epoch", None)
            phase = getattr(collapse_pipeline, "last_incremental_collapse_phase", None)
            current_frame_id = int(getattr(engine, "frame_id", frame_index + 1))
            epoch_id = int(
                getattr(epoch, "epoch_id", getattr(collapse_pipeline, "incremental_collapse_epoch_sequence", 0))
            )
            started_frame_id = int(
                getattr(
                    epoch,
                    "started_frame_id",
                    getattr(collapse_pipeline, "last_incremental_collapse_epoch_started_frame_id", current_frame_id),
                )
                or current_frame_id
            )
            frame_samples.append(
                {
                    "frame_id": current_frame_id,
                    "frame_mod4": current_frame_id % 4,
                    "frame_ms": elapsed_ms,
                    "collapse_phase": None if phase is None else int(phase),
                    "epoch_id": epoch_id,
                    "epoch_age": max(0, current_frame_id - started_frame_id),
                    "epochs_started": int(getattr(collapse_pipeline, "incremental_collapse_epochs_started", 0)),
                    "epochs_completed": int(getattr(collapse_pipeline, "incremental_collapse_epochs_completed", 0)),
                    "outstanding": epoch is not None,
                }
            )
            if profile_passes:
                pass_profiles.append(dict(engine.last_pass_profile))
            readbacks_completed += len(engine.poll_all_readbacks())

        report = engine.simulation_backend_report()
        result: dict[str, object] = {
            "scenario": name,
            "frames": frames,
            "avg_ms": statistics.fmean(frame_ms) if frame_ms else 0.0,
            "min_ms": min(frame_ms, default=0.0),
            "p95_ms": percentile(frame_ms, 0.95),
            "max_ms": max(frame_ms, default=0.0),
            "frame_samples": frame_samples,
            "readbacks_completed": readbacks_completed,
            "backend_report": report,
            "heat_terminal_phase_fusion": active_heat_terminal_phase_fusion,
            "heat_terminal_dirty_publish_fusion": active_heat_terminal_dirty_publish_fusion,
            "heat_terminal_workgroup16x8": active_heat_terminal_workgroup16x8,
        }
        phase_frame_ms = {
            phase: [
                float(sample["frame_ms"])
                for sample in frame_samples
                if sample["collapse_phase"] == phase
            ]
            for phase in range(4)
        }
        result["collapse_phase_frame_ms"] = {
            str(phase): {
                "count": len(values),
                "avg_ms": statistics.fmean(values) if values else 0.0,
                "p95_ms": percentile(values, 0.95),
                "max_ms": max(values, default=0.0),
            }
            for phase, values in phase_frame_ms.items()
        }
        if profile_passes:
            result["pass_profiles"] = pass_profiles
            result["pass_profile_summary"] = summarize_pass_profiles(pass_profiles)
            result["skipped_gpu_stages"] = list(report.get("gpu_realtime_budget", {}).get("skipped_stages", []))
        return result
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DustWhisper core world engine GPU scenarios.")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--readback", action="store_true", help="Queue one small async readback per frame.")
    parser.add_argument("--profile-passes", action="store_true", help="Record per-frame pass-level CPU timings.")
    parser.add_argument(
        "--profile-passes-sync",
        action="store_true",
        help="Synchronize around profiled passes to attribute queued GPU work to the pass that launched it.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--heat-terminal-phase-fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Benchmark terminal4x6 phase/boil recomputation without implying dirty publish fusion.",
    )
    parser.add_argument(
        "--heat-terminal-dirty-publish-fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Benchmark terminal4x6 dirty queue publication without implying phase/boil fusion.",
    )
    parser.add_argument(
        "--heat-terminal-workgroup16x8",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Benchmark the default-off terminal4x6 16x8 workgroup variant.",
    )
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), help="Run only the selected scenario; repeatable.")
    args = parser.parse_args()

    ctx, backend_label = create_context()
    info = getattr(ctx, "info", {})
    selected = args.scenario or list(SCENARIOS)
    results = [
        run_scenario(
            name,
            SCENARIOS[name],
            ctx=ctx,
            width=args.width,
            height=args.height,
            warmup=args.warmup,
            frames=args.frames,
            dt=args.dt,
            readback=args.readback,
            profile_passes=args.profile_passes,
            profile_passes_sync=args.profile_passes_sync,
            heat_terminal_phase_fusion=args.heat_terminal_phase_fusion,
            heat_terminal_dirty_publish_fusion=args.heat_terminal_dirty_publish_fusion,
            heat_terminal_workgroup16x8=args.heat_terminal_workgroup16x8,
        )
        for name in selected
    ]
    payload = {
        "context_backend": backend_label,
        "gpu": {
            "renderer": info.get("GL_RENDERER", ""),
            "vendor": info.get("GL_VENDOR", ""),
            "opengl_version": info.get("GL_VERSION", ""),
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Context backend: {backend_label}")
    print(f"Renderer: {payload['gpu']['renderer']}")
    print(f"Vendor: {payload['gpu']['vendor']}")
    print(f"OpenGL: {payload['gpu']['opengl_version']}")
    for result in results:
        report = result["backend_report"]
        assert isinstance(report, dict)
        non_gpu = report.get("non_gpu_backends", {})
        print(
            f"{result['scenario']}: avg={result['avg_ms']:.3f}ms "
            f"p95={result['p95_ms']:.3f}ms max={result['max_ms']:.3f}ms "
            f"strict_gpu_ready={report.get('strict_gpu_ready')} non_gpu={non_gpu}"
        )
        if args.profile_passes:
            skipped = result.get("skipped_gpu_stages", [])
            print(f"  skipped_gpu_stages={skipped}")
            summary = result.get("pass_profile_summary", {})
            if isinstance(summary, dict):
                for name, values in summary.items():
                    if isinstance(values, dict):
                        avg_gpu = values.get("avg_gpu_ms")
                        line = f"  {name}: avg_cpu={float(values.get('avg_cpu_ms', 0.0)):.3f}ms"
                        if avg_gpu is not None:
                            line += f" avg_gpu={float(avg_gpu):.3f}ms"
                        line += f" count={values.get('count')}"
                        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
