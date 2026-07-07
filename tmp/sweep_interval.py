import sys, statistics, time
sys.path.insert(0, '.')
import scripts.benchmark_engine as B
import moderngl
from oracle_game.world import WorldEngine

def run(interval, frames=20, warmup=3):
    ctx, _ = B.create_context()
    eng = WorldEngine(width=1920, height=1080, gpu_context=ctx)
    eng.formal_collapse_interval_frames = interval
    B.setup_random_materials(eng)
    eng.bridge.sync_world(eng, force_cpu_resource_upload=True); eng._gpu_cpu_dirty_resources.clear()
    pw = getattr(eng, "prewarm_formal_connected_collapse", None)
    if callable(pw): pw()
    for _ in range(warmup):
        eng.step(1/60); ctx.finish(); eng.poll_all_readbacks()
    ms=[]
    for _ in range(frames):
        s=time.perf_counter(); eng.step(1/60); ctx.finish(); ms.append((time.perf_counter()-s)*1000)
        eng.poll_all_readbacks()
    eng.close()
    return ms

for interval in [4, 6, 8, 12]:
    ms = run(interval)
    print(f'interval={interval}: avg={statistics.fmean(ms):.2f} p95={sorted(ms)[int(len(ms)*0.95)-1]:.2f} min={min(ms):.2f} max={max(ms):.2f}')
