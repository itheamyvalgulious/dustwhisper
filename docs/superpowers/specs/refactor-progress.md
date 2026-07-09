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
- [x] Phase 1 (GPU base) — `sim/gpu_base.py` `GPUPipelineBase`; all 11 GPU pipelines subclass it (dup helpers removed; `_barrier_bits` overrides where needed). Snapshot unchanged.
- [~] Phase 1 (other foundations) — shader loader built (`sim/shader_loader.py`); `EngineConfig`, `types/` split, `cpu/base`, `MaterialTableAccessor` still TODO.
- [~] Phase 2 (shader extraction) — pilot done: gpu_merge GLSL → `shaders/merge/merge_cell_core.comp` (loader verified source-identical; gpu_merge 295→115 lines). 277 shaders remain across 10 files.
- [~] Phase 4 (world.py split) — in progress:
  - [x] constants block → `world_constants.py`
  - [x] `serialize_engine_capabilities` (4254 lines) → `world_capabilities.py` (world.py 17279→12948)
  - [ ] `world_capabilities.py` split into <1000 sections (subagent running)
  - [ ] `_make_readback_payload` → `world_readback_payload.py` (subagent running)
  - [ ] geometry bucket, intent-resolution cluster, input coercion, debug-frame, payload serializers (TODO)

## Files still > 1000 lines (after current work)
world.py (12948), gpu_collapse (11347), gpu_reactions (9675), gpu_motion (9210),
gpu_liquid (4775), gpu.py (4465), world_capabilities.py (4284, being split),
gpu_heat (2246), reactions.py (2011), rules.py (1578), enginedemo.py (1463),
motion.py (1417), gpu_optics (1273), http_console.py (1164), gpu_gas (1095),
gpu_page_stripes (1073), gpu_world_commands (1041), liquid.py (1011).

## Per-extraction gates (in addition to snapshot)
- capabilities golden: `f65d2183375bd352`
- readback golden: `7062d287b034df0c`
- GPU snapshot: `ce71a34376c5010d`
