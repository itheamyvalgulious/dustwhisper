# Comprehensive Structural Refactor — Design Spec

Date: 2026-07-09
Branch: `refactor/structure`
Scope: `oracle_game/` (~78K lines) — behavior-preserving restructure.

## 1. Goals

- Keep all existing logic/numerics/stage behavior identical (AGENTS.md: 不丢质量; no reduction of density/resolution/active-region/solver coverage/visual fidelity).
- Reduce code complexity: factor duplicated code into shared base classes/helpers.
- Fully separate CPU and GPU code into distinct packages.
- Separate update stages: uniform lifecycle via base class + registry, data-driven dispatch.
- Separate configuration into one central `EngineConfig`.
- **Hard constraint: every source file ≤ 1000 lines.**
- Add appropriate comments throughout.

## 2. Current state (problems)

- `world.py` (17,279 lines): god-class `WorldEngine` (~270 methods, 24 responsibility buckets) holding *all* state — cell/gas/optics arrays, ~30 parallel `material_*` numpy tables, GPU handles, config thresholds, readback/entity state. Contains a single 4,254-line method `serialize_engine_capabilities`. CPU↔GPU split unclean: one `simulation_backend` flag drives ~30 inline branches.
- 7 CPU solvers + 11 GPU pipelines, **no base class, no registry**. Stages hardcoded in 3 places in `world.py` (ctor L353-359, reset L1891-1897, step-loop L4660-4759).
- Heavy copy-paste: `_formal_gpu_frame` byte-identical in 9 files; `_sync_compute_writes` near-identical in 9; `_set_uniform_if_present`/`_write_dynamic_buffer`/`available`/`_profile_pass`/`_ensure_resources`-prologue/`release` all duplicated; `_material_table_row` verbatim in 3 CPU solvers; `_masked_range` verbatim in 2.
- ~50% of each big GPU file is GLSL embedded as inline f-strings (gpu_collapse ≈ 5900 GLSL + 5400 Python).
- Config scattered as inline `__init__` attrs + module constants — no central object.

## 3. Target package layout

```
oracle_game/
  config/
    engine_config.py        # EngineConfig dataclass (all thresholds/sizes/budgets)
    defaults.py             # default constants
  types/                    # enums.py content.py commands.py entities.py frame.py
  rules.py + content_defaults.py   # RuleBook | default _build_* tables
  state/
    world_state.py          # WorldState: owns grid/gas/optics arrays
    cell_state.py gas_state.py optics_state.py
    material_tables.py      # ~30 material_* arrays + MaterialTableAccessor
  cpu/                      # CPU ONLY (no moderngl, no GPUBridge imports)
    base.py                 # Solver base: lifecycle + _select_backend
    registry.py             # ordered stage registry → data-driven dispatch
    material_accessor.py active_region.py field_utils.py
    collapse.py gas.py heat.py liquid.py motion.py optics.py
    reactions.py (+ reactions_pairings.py if >1000)
  gpu/                      # GPU ONLY (no CPU sim logic)
    base.py                 # GPUPipeline base (absorbs 8 duplicated helpers)
    bridge.py dtypes.py packers.py readback.py dirty_tiles.py
    _shader_loader.py       # loads shaders/, substitutes {{constants}}, caches
    pipeline_collapse.py pipeline_reactions.py pipeline_motion.py pipeline_liquid.py
    pipeline_heat.py pipeline_optics.py pipeline_gas.py
    pipeline_merge.py pipeline_placeholders.py pipeline_page_stripes.py pipeline_world_commands.py
  shaders/                  # pure .glsl, per-pass; common/ for shared preambles
  engine/                   # orchestrator (world.py 17K → focused modules, each <1000)
    world_engine.py         # thin: owns state+config+bridge; delegates to stages + modules
    frame_pipeline.py       # step/run_cpu_frame/_step_once_impl (data-driven stage loop)
    input_coercion.py table_validation.py table_api.py command_queue.py
    frame_io.py paging_api.py entity_sync.py controller_turn.py
    capabilities_schema.py  # 4254-line monster → data table
    bridge_serializer.py payload_serializer.py debug_frame.py
    readback_builder.py shadow_tables.py intent_resolver.py geometry.py
    observation.py runtime_rebuild.py demo_scene.py
  http_console.py  enginedemo.py (+ demo_input.py/demo_render.py/demo_controller.py)
```

## 4. Key abstractions

1. **`Solver` base (`cpu/base.py`)** — 5-method lifecycle (`step`/`reset_runtime_state`/`release`/`runtime_snapshot`) + `_select_backend(world, stage)` absorbing the duplicated "formal-GPU-frame / active-scheduler-authoritative" prologue.
2. **`GPUPipeline` base (`gpu/base.py`)** — absorbs `_formal_gpu_frame`, `_sync_compute_writes` (overridable `_barrier_bits`), `_set_uniform_if_present`, `_write_dynamic_buffer`, `available`, `_profile_pass`, `_ensure_resources` prologue, `release`. Subclasses override `_resource_signature`/`_build_resources`/`_build_programs`/entry.
3. **`EngineConfig`** — single dataclass; `WorldEngine.__init__` takes it; solvers/pipelines read `world.config`.
4. **`MaterialTableAccessor`** — one home for `_material_table_row`/`_material_*` helpers.
5. **Stage registry** — `_step_once_impl` loop (collapse→gas→heat→reactions×6→motion→liquid→optics→latch_clear→active_decay) becomes data-driven; adding a stage = one line.
6. **Shader loader** — replaces inline f-string GLSL; `.glsl` + `{{MAX_MATERIALS}}`-style substitution from config.

## 5. CPU/GPU separation rule

`cpu/` imports no `moderngl`/`GPUBridge`; `gpu/` holds no CPU sim logic. Each CPU `Solver` keeps an optional `gpu_pipeline` handle and delegates in `step()` via `_select_backend` — preserving today's dual-backend behavior exactly. `WorldEngine` owns `GPUBridge` + the 4 frame-level pipelines (merge/placeholder/page_stripe/world_commands).

## 6. God-class split approach

**Approach A (chosen): split `world.py` by responsibility** into `engine/*` modules matching the existing 24 buckets — low rewiring, low parity risk.
Alternative B (split by carrier CellWorld/GasWorld/OpticsWorld) deferred — most methods are cross-cutting; revisit once seams exist.

## 7. Phasing (each phase behavior-preserving; verified before next)

- **Phase 0** — verification harness: confirm `tests/test_engine_core.py` green; lock a frame-parity snapshot (CPU-oracle vs GPU-world) as golden.
- **Phase 1** — foundation: `EngineConfig`, `types/` split, `cpu/base.py` + `gpu/base.py`, `MaterialTableAccessor`, shader loader. No big moves yet.
- **Phase 2** — GPU pipelines onto base + GLSL → `shaders/`; split the 4 giants (collapse/reactions/motion/liquid) per-pass.
- **Phase 3** — CPU solvers onto base + registry; data-driven dispatch.
- **Phase 4** — `world.py` god-class split bucket-by-bucket (`capabilities_schema` first).
- **Phase 5** — entry-point splits; final <1000 audit.

## 8. Verification gate

After every phase: run `tests/test_engine_core.py` + parity/snapshot scripts (`tmp/behavior_snapshot.py` etc.) + frame-by-frame CPU↔GPU parity check. A phase advances only when parity holds. If parity breaks, revert and fix before proceeding.

## 9. Non-goals

- No new features, no algorithm changes, no perf tuning (perf work lives on `perf/*` branches).
- No changing public API surface of `WorldEngine`/`enginedemo` entry points beyond import-path moves.
- No re-enabling the disabled Phase C merge (stays disabled).
