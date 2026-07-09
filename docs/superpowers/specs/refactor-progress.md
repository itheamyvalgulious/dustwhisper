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
  - [x] gpu_reactions (9675→5148), gpu_heat (2246→1056), gpu_optics (1273→635), gpu_world_commands (1041→705) — shaders extracted (includes mechanism for preambles). ALL 10 GPU shader migrations complete; 8 of 10 now < 1000.
  - Note: compound f-string expressions (`{LOCAL_SIZE-1}`, `{A*B}`, `{Cls.ATTR}`) → derived markers (motion/reactions/heat/optics/world_commands/liquid/placeholders) or baked literals (some collapse .comp — robustness follow-up).
- [~] Phase 4 (world.py split) — world.py 17279→10819 (7 extractions done):
  - [x] constants → `world_constants.py`
  - [x] `serialize_engine_capabilities` → `world_capabilities/` package (12 modules, each ≤916 lines)
  - [x] `_make_readback_payload` → `world_readback_payload.py`
  - [x] `debug_frame` + 15 frame helpers → `world_debug_frame.py` (502 lines)
  - [x] geometry bucket (34 methods) → `world_geometry.py` (679 lines; golden `d9e44e209feaedd6`)
  - [x] `serialize_*_runtime` (9 methods) → `world_runtime_serializers.py` (489 lines; golden `34f8fec1267d53a2`)
  - [x] bridge serializers (21 methods) → `world_bridge_serializers.py` (621 lines; golden `7749626c1398fd35`)
  - [ ] payload serializers (`serialize_*` scattered), intent-resolution cluster (interconnected), input coercion (`_coerce_*`/`_normalize_*`), shadow/sanctioned tables, runtime rebuild, controller-turn, paging API (TODO; same verbatim-move + golden pattern)
- [x] gpu.py (4465) → `gpu/` package: `_common` (60), `dtypes` (587), `packers` (1186), `readback` (142), `bridge` (2723=GPUBridge), `__init__` (32). All `from oracle_game.gpu import X` paths preserved.

## Files still > 1000 lines (current)
world.py (10819), gpu_collapse (5519), gpu_reactions (5148), gpu_motion (3367),
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

## Session summary (2026-07-09) — 29 commits on `refactor/structure`

### Accomplished (all behavior-preserving, verified via snapshot + goldens)
- **Phase 1 (GPU base):** `sim/gpu_base.py` `GPUPipelineBase` (available/reset_pass_profile/_profile_pass/_formal_gpu_frame/_set_uniform_if_present/_sync_compute_writes + `_barrier_bits` hook). All 11 GPU pipelines subclass it; ~6 duped helpers removed each.
- **Phase 1 (CPU):** `sim/cpu_base.py` shared `material_table_row` (3 solvers delegate). `Solver` base + `_select_backend` + stage registry pending (Phase 3).
- **Phase 2 (shaders):** `sim/shader_loader.py` (template `{{NAME}}` substitution + `includes` for preambles). ALL 10 GPU pipelines' GLSL extracted to `oracle_game/shaders/<stage>/*.comp` (~288 shaders): gpu_merge(115), gpu_page_stripes(868), gpu_placeholders(346), gpu_gas(591), gpu_optics(635), gpu_world_commands(705), gpu_heat(1056), gpu_liquid(1542), gpu_motion(3370), gpu_collapse(5519), gpu_reactions(5148). Independent verifier `scripts/verify_shaders.py` (0 failures). Compound f-string exprs → derived markers (or baked literals in some collapse files — robustness follow-up).
- **Phase 2/4 (gpu.py):** 4465 → `gpu/` package (_common/dtypes/packers/readback/bridge + __init__ re-exports). packers(1186) & bridge(2723) still >1000.
- **Phase 4 (world.py):** 17279 → 11682. Extracted: constants → `world_constants.py`; `serialize_engine_capabilities` → `world_capabilities/` 12-module package; `_make_readback_payload` → `world_readback_payload.py`; `debug_frame`+15 → `world_debug_frame.py`; geometry bucket (34 methods) → `world_geometry.py`; runtime serializers → `world_runtime_serializers.py` (in progress).
- **Bug found & fixed:** the gpu_liquid migration's verification was flawed — 32 `{EXPR}` remnants (`{MAX_MATERIALS - 1}` etc.) across 17 liquid .comp + 1 in placeholders rendered as invalid GLSL. Caught by the independent verifier; fixed via derived markers. The snapshot's scenario didn't compile those shaders (latent), but the 96×64 debug scenario did.

### Key methodology (reusable for continuation)
- **Gate = deterministic golden hash per extraction** (snapshot `ce71a34376c5010d` for the sim; per-bucket goldens for non-sim methods) + `scripts/verify_shaders.py` for all .glsl. Capture the golden BEFORE the move; the move is a verbatim `self`→`engine` lift, so a matching golden proves preservation.
- **Subagent pattern:** dispatch `code-simplifier` per file/bucket with the exact golden + snapshot gate in the prompt; the verbatim-move + golden-check is mechanical and parallelizable across disjoint files.

### Remaining work (prioritized)
1. **world.py (11682 → <1000):** extract remaining buckets — payload serializers (`serialize_*`, scattered ~L2779-6900), intent-resolution cluster (~L9748-11000, interconnected), input coercion (`_coerce_*`/`_normalize_*` ~995 lines), shadow/sanctioned tables, bridge serializers, runtime rebuild, controller-turn, paging API, etc. Same verbatim-move + golden pattern.
2. **Pipeline logic splits (still >1000):** gpu_collapse(5519), gpu_reactions(5148), gpu_motion(3370), gpu_liquid(1542), gpu_heat(1056), gpu/packers(1186), gpu/bridge(2723). These are non-shader Python (resource dataclasses + dispatch/stage methods). Split each pipeline class into focused modules (resources / stages / publish) — harder, needs per-pipeline care; gate via snapshot.
3. **CPU solver splits:** reactions.py(2011), motion.py(1413), liquid.py(1007). Split each solver's logic (e.g. reactions.py pairings → separate module). Gate via a CPU snapshot golden (capture one).
4. **rules.py (1578):** `_build_materials` is itself ~1022 lines (a single function) — split into per-category sub-builders composed in `_build_materials`. Gate via `RULES_GOLDEN=5d8c712ed57c4a46`.
5. **Entry points:** enginedemo(1463) → main + demo_input/demo_render/demo_controller; http_console(1164) → split handlers. Verify via import + a headless run if possible.
6. **Phase 1 leftovers:** `EngineConfig` (centralize the scattered config constants/thresholds); `types/` split (types.py is 560, under 1000 — low priority); package reorg (rename `sim/`→`cpu/`+`gpu/`, `engine/` for the world.py-extracted modules) with re-export shims; `Solver` base + stage registry (Phase 3) to make `_step_once_impl` data-driven.
