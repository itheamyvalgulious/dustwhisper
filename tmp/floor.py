import sys, statistics, time
sys.path.insert(0, '.')
import scripts.benchmark_engine as B
from oracle_game.world import WorldEngine

ctx, _ = B.create_context()
eng = WorldEngine(width=1920, height=1080, gpu_context=ctx)
B.setup_random_materials(eng)
eng.bridge.sync_world(eng, force_cpu_resource_upload=True); eng._gpu_cpu_dirty_resources.clear()
pw = getattr(eng, "prewarm_formal_connected_collapse", None)
if callable(pw): pw()
for _ in range(4): eng.step(1/60); ctx.finish(); eng.poll_all_readbacks()
ms=[]
for i in range(24):
    s=time.perf_counter(); eng.step(1/60); ctx.finish(); ms.append((time.perf_counter()-s)*1000)
    eng.poll_all_readbacks()
eng.close()
ms_sorted=sorted(ms)
print(f'avg={statistics.fmean(ms):.2f} min={min(ms):.2f} max={max(ms):.2f}')
print(f'non-collapse floor (min): {min(ms):.2f}ms — this is the best possible single-frame time')
print(f'collapse-frame cost (max-min): {max(ms)-min(ms):.2f}ms spike on collapse frames')
print(f'For avg=60 with 1-in-4 collapse spike: non-collapse must be {(60*4-max(ms))/3:.1f}ms (reactions alone ~32ms)')
