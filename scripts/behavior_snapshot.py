from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oracle_game.world import WorldEngine
from scripts.benchmark_engine import create_context, setup_random_materials


SNAPSHOT_FIELDS = (
    "material_id",
    "phase",
    "cell_temperature",
    "integrity",
    "velocity",
    "gas_concentration",
    "ambient_temperature",
    "timer_pack",
    "island_id",
    "entity_id",
)


def snapshot(
    *,
    frames: int = 20,
    warmup: int = 3,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, np.ndarray]:
    ctx, _ = create_context()
    engine = WorldEngine(width=width, height=height, gpu_context=ctx)
    try:
        setup_random_materials(engine)
        engine.bridge.sync_world(engine, force_cpu_resource_upload=True)
        engine._gpu_cpu_dirty_resources.clear()
        engine.prewarm_formal_connected_collapse()
        for _ in range(max(0, warmup)):
            engine.step(1.0 / 60.0)
            ctx.finish()
            engine.poll_all_readbacks()
        for _ in range(max(0, frames)):
            engine.step(1.0 / 60.0)
            ctx.finish()
            engine.poll_all_readbacks()
        return {
            name: np.ascontiguousarray(getattr(engine, name)).copy()
            for name in SNAPSHOT_FIELDS
        }
    finally:
        engine.close()


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("ascii"))
        digest.update(arrays[name].tobytes())
    return digest.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash a deterministic GPU simulation snapshot.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    for trial in range(max(1, args.trials)):
        started = time.perf_counter()
        arrays = snapshot(
            frames=args.frames,
            warmup=args.warmup,
            width=args.width,
            height=args.height,
        )
        field_hashes = {
            name: hashlib.sha256(value.tobytes()).hexdigest()[:8]
            for name, value in sorted(arrays.items())
        }
        elapsed = time.perf_counter() - started
        print(
            f"trial {trial}: hash={hash_arrays(arrays)} ({elapsed:.1f}s) "
            f"per-array={field_hashes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
