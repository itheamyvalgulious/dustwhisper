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
- [x] Phase 1 (GPU base) — `sim/gpu_base.py` `GPUPipelineBase`; all 11 GPU pipelines subclass it (dup helpers removed; `_barrier_bits` overrides where needed).
- [~] Phase 1 (other foundations) — shader loader built (`sim/shader_loader.py`, with `includes`); shared `material_table_row` extracted to `sim/cpu_base.py` (3 CPU solvers delegate). `EngineConfig`, `types/` split, `Solver` base + `_select_backend` still TODO (Phase 3).
- [~] Phase 2 (shader extraction):
  - [x] gpu_merge (295→115), gpu_page_stripes (1073→868), gpu_placeholders (657→346), gpu_gas (1095→591) — fully migrated, < 1000.
  - [x] gpu_motion (9210→3370), gpu_liquid (4775→1540), gpu_collapse (11347→5522) — shaders extracted; STILL > 1000 (remaining is non-shader pipeline Python logic; further split pending).
  - [ ] gpu_reactions (9675), gpu_heat (2246), gpu_optics (1273), gpu_world_commands (1041) — preamble files, TODO (use `includes`).
  - Note: compound f-string expressions (`{LOCAL_SIZE-1}`, `{A*B}`, `{Cls.ATTR}`) baked as evaluated literals in some collapse .comp (behavior-identical; derived-marker conversion is a robustness follow-up).
- [~] Phase 4 (world.py split) — world.py 17279→12093 (geometry extraction in progress):
  - [x] constants → `world_constants.py`
  - [x] `serialize_engine_capabilities` → `world_capabilities/` package (12 modules, each ≤916 lines)
  - [x] `_make_readback_payload` → `world_readback_payload.py`
  - [x] `debug_frame` + 15 frame helpers → `world_debug_frame.py` (502 lines)
  - [~] geometry bucket → `world_geometry.py` (subagent running; golden `d9e44e209feaedd6`)
  - [ ] intent-resolution cluster, input coercion, payload serializers (TODO; intent cluster is large/interconnected)
- [x] gpu.py (4465) → `gpu/` package: `_common` (60), `dtypes` (587), `packers` (1186), `readback` (142), `bridge` (2723=GPUBridge), `__init__` (32). All `from oracle_game.gpu import X` paths preserved.

## Files still > 1000 lines (current)
world.py (12093), gpu_collapse (5519), gpu_reactions (5148), gpu_motion (3367),
gpu/bridge (2723=GPUBridge class), reactions.py (2011, CPU), rules.py (1578),
gpu_liquid (1542), enginedemo (1463), motion.py (1413, CPU), gpu/packers (1186),
http_console (1164), gpu_heat (1056), liquid.py (1007).
(Total: ~51800 lines, down from 78245. ~26500 lines extracted to .glsl files +
focused modules.) Remaining work: world.py more extractions; pipeline LOGIC
splits (gpu_collapse/reactions/motion/liquid/heat — non-shader Python); CPU
solver splits (reactions/motion/liquid); rules _build_materials split;
enginedemo/http_console splits; EngineConfig; types/ split; package reorg
(cpu/, gpu/ done-partial, engine/); Solver base + stage registry (Phase 3).

## Per-extraction gates (in addition to snapshot)
- capabilities golden: `f65d2183375bd352`
- readback golden: `7062d287b034df0c`
- debug_frame golden: `b4b5996932795cbd` (DebugView×all + gas views, 96×64, 3 frames). NOTE: an earlier value (b4e60f4b007fe5a0) was captured AFTER the buggy liquid migration and reflected broken-liquid behavior; the liquid remnant fix corrected it to this true value.
- geometry golden: `d9e44e209feaedd6` (7 coord methods on a 96×64 populated world)
- GPU snapshot: `ce71a34376c5010d`
