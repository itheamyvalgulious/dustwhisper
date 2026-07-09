# Structural Refactor — Progress Log

Baseline captured 2026-07-09 on branch `refactor/structure` (off `perf/step2-phase-c`).

## Verification gate (must hold after every phase)

1. **CPU-path unit tests stay green:** `.venv/bin/python -m pytest tests/ -x -q` → 30 passing.
2. **Known pre-existing GPU-parity failure stays identical:** `test_patch_material_changes_friction_motion_behavior_in_cpu_and_gpu_paths` fails on `master` too (cpu vy=54.4, gpu vy=0.0). Behavior-preserving refactor ⇒ failure signature must not change.
3. **GPU behavior snapshot hash unchanged:** `tmp/behavior_snapshot.py`, 480×270, 20 frames, warmup 3 → `ce71a34376c5010d`.

Re-run after each phase:
```
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -c "import sys;sys.path.insert(0,'.');from tmp.behavior_snapshot import snapshot,hash_arrays as h;print(h(snapshot(frames=20,warmup=3,width=480,height=270)))"
```

## Phases
- [x] Phase 0 — baseline captured.
- [ ] Phase 1 — foundation (EngineConfig, types/ split, cpu/base, gpu/base, MaterialTableAccessor, shader loader).
  - [x] 1.1 `sim/gpu_base.py` — `GPUPipelineBase` (available, reset_pass_profile, _profile_pass, _formal_gpu_frame, _set_uniform_if_present, _sync_compute_writes + _barrier_bits hook). Pilot: `gpu_merge` subclassed (3 dup methods removed).
- [ ] Phase 2 — GPU pipelines onto base + GLSL → shaders/.
- [ ] Phase 3 — CPU solvers onto base + registry.
- [ ] Phase 4 — world.py god-class split.
- [ ] Phase 5 — entry-point splits + <1000 audit.
