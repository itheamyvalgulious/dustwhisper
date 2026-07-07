import sys, hashlib
sys.path.insert(0, '.')
import numpy as np
import scripts.benchmark_engine as B
from oracle_game.world import WorldEngine

def snapshot(frames=20, warmup=3, width=1920, height=1080):
    ctx, _ = B.create_context()
    eng = WorldEngine(width=width, height=height, gpu_context=ctx)
    B.setup_random_materials(eng)
    eng.bridge.sync_world(eng, force_cpu_resource_upload=True); eng._gpu_cpu_dirty_resources.clear()
    pw = getattr(eng, "prewarm_formal_connected_collapse", None)
    if callable(pw): pw()
    for _ in range(warmup): eng.step(1/60); ctx.finish(); eng.poll_all_readbacks()
    for _ in range(frames): eng.step(1/60); ctx.finish(); eng.poll_all_readbacks()
    # download GPU-authoritative state for comparison
    arrays = {}
    for name in ("material_id","phase","cell_temperature","integrity","velocity","gas_concentration","ambient_temperature","timer_pack","island_id","entity_id"):
        a = getattr(eng, name, None)
        if a is None: continue
        arrays[name] = np.ascontiguousarray(a).copy()
    eng.close()
    return arrays

def hash_arrays(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(name.encode()); h.update(arrays[name].tobytes())
    return h.hexdigest()[:16]

if __name__ == "__main__":
    import time
    for trial in range(2):
        t=time.perf_counter()
        a = snapshot()
        print(f'trial {trial}: hash={hash_arrays(a)} ({time.perf_counter()-t:.1f}s) per-array:', {n:hashlib.sha256(a[n].tobytes()).hexdigest()[:8] for n in sorted(a)})
