# GPU Simulation Performance Optimization Change Log

This document records the production changes included in the 2026-07 GPU
simulation optimization commit. It separates enabled optimizations from
fallback and experimental implementations so benchmark claims can be traced to
the actual default runtime path.

## Scope And Result

- Target workload: `1920x1080`, `random_materials`, full active region, all GPU
  solvers enabled, no budgeted stage skips.
- Reference run before the retained optimization series: average `27.9305ms`,
  P95 `29.8425ms`, maximum `30.7236ms` on a GeForce GTX 1650.
- Final validation run O1: average `14.5246ms`, P95 `14.8493ms`, maximum
  `16.0460ms` over 512 measured frames.
- Final validation run O2: average `14.5360ms`, P95 `14.8747ms`, maximum
  `15.9784ms` over 512 measured frames.
- A later 256-frame confirmation run measured average `14.5862ms`, P95
  `15.0015ms`, and maximum `16.5852ms`. This confirms 60 FPS average
  throughput but shows that the strict `16.3ms` maximum-frame target is not yet
  guaranteed on every shorter rerun.
- Both final runs reported `strict_gpu_ready=true`; collapse, gas, heat,
  reactions, motion, liquid, and optics used GPU backends; skipped stages were
  empty.
- The result does not reduce the material grid, active region, solver coverage,
  rule coverage, or visual fidelity. The existing `gas_cell_size=4` model is
  unchanged.

## Repository And Verification Tooling

- Removed the tracked and untracked contents of `tmp/`: benchmark JSON,
  synchronized profiles, raw snapshots, exact-check wrappers, logs, error
  excerpts, shell wrappers, and one-off experiment scripts.
- Removed the root `progress.md` debug journal. Git history retains it if an old
  investigation must be recovered.
- Added scoped `tmp/*` ignore rules while retaining `tmp/.gitkeep`, so the
  project-local scratch directory required by `AGENTS.md` remains available.
- Added ignore rules for coverage output and common analysis caches.
- Promoted the deterministic GPU snapshot utility from
  `tmp/behavior_snapshot.py` to `scripts/behavior_snapshot.py`, added a stable
  CLI, guaranteed engine cleanup, and updated structural-refactor documents.
- Removed a unit test that inspected a temporary checker file as source text.
  Production generation lifecycle tests remain in the test module.
- Extended `scripts/benchmark_engine.py` with deterministic random-material
  setup, raw frame samples, collapse phase attribution, strict backend reports,
  optional synchronized pass profiling, and explicitly gated heat candidates.

## Shared GPU Data Flow And Frame Scheduling

- Added a lazily allocated private `cell_core` spare buffer to `GPUBridge` for
  liquid provenance ping-pong without changing the authoritative public buffer
  name or readback lookup.
- Added a private compact powder-reservation buffer and materialization hooks so
  normal GPU frames do not maintain the public 48-byte reservation ABI. Debug,
  serialization, and CPU upload paths expand it on demand.
- Invalidated optics sparse runtime when CPU uploads overwrite optical state.
- Preserved GPU-authoritative ownership across solver boundaries to avoid CPU
  downloads, repeat uploads, and full-grid bridge hydration.
- Added same-frame heat-to-reaction-to-motion handoff state. Later stages consume
  producer textures directly and publish the final cell core once.
- Added exception recovery for deferred heat publication so a failed later
  stage restores the last valid post-heat bridge state.
- Added reaction-to-motion terminal integration and fused reaction-latch clear.
  Motion semantics still execute; only redundant hydration/integration passes
  are removed.
- Kept the broad one-frame-latency Phase-C merge disabled because it changed
  condensation position despite its performance benefit.
- Kept `_skip_budgeted_gpu_stage` disabled. Performance results therefore do not
  omit solver stages.
- Added shared packed timer and packed cell-state helpers used by multiple
  pipelines and parity tests.

## Collapse

- Added `FormalDirtyCollapseEpoch`, which owns support textures, label state,
  schedules, dirty snapshots, and commit state across four frames.
- Claimed the dirty queue at epoch start; changes produced during the epoch
  remain queued for the next epoch instead of mutating in-flight inputs.
- Added world/resource signatures. Resize, paging, material-generation, context,
  or resource changes abort the stale epoch and request a full rebuild.
- Rebalanced the epoch: phase 0 performs classification and two support jumps;
  phase 1 performs remaining coarse jumps; phase 2 performs refine, outcome,
  and label union; phase 3 validates and materializes.
- Spread compact island-runtime publication across four admission slots after
  the main commit.
- Added persistent dense tile worklists keyed by context, geometry, region, and
  buffer identity. Frontier mutation and resource teardown invalidate them.
- Fused bridge hydration, structural classification, support seed generation,
  runtime masks, pending state, and row/column axis masks.
- Added full-resolution `R8UI` support ping-pong textures. The canonical float
  path remains available as a fallback.
- Replaced scalar per-cell row propagation with 32-bit row masks, bit-shift
  horizontal closure, shared vertical closure, and row/column segment tests.
- Added row-major support output and propagated-source structural-mask elision.
  Connected-tile checks still reject stale texels.
- Added an NV32 specialization. Each warp lane preloads one structural row,
  `ballotThreadNV` constructs support masks, `gl_ThreadInWarpNV` supplies the
  ballot bit index, and explicit grid checks protect partial tiles.
- Runtime-probed the required NV extensions and actual warp size. Unsupported
  devices use the canonical U8 shader.
- Replaced full-resolution component-label JFA with per-tile local components,
  cross-tile boundary edges, adaptive union-find, and final materialization.
- Fused outcome resolution with local label-root initialization.
- Fused label-union materialization with live-state validation.
- Fused component metadata reduction and filtered-label publication into the
  final bridge materialization dispatch.
- Replaced per-epoch full component-invalid clears with a 32-bit generation;
  wrap performs the required physical clear.
- Replaced root-flag clears with the same generation validity model and filters
  stale flags during materialization.
- Added direct immune and delayed outcome publication plus stride-based runtime
  admission dispatch.
- Added partial-tile, disconnected-stale-texel, two-epoch, mutation, generation
  wrap, tile-union, materialization, and NV32 regression tests.

## Gas

- Fused divergence construction with the first pressure Jacobi seed and removed
  pressure initialization that was immediately overwritten.
- Added a two-iteration Jacobi shader. It computes the first iteration into a
  shared halo, synchronizes, computes the second iteration, and preserves the
  odd tail iteration and canonical outputs.
- Added a density SSBO path and shared tree reduction that preserves the legacy
  pairwise floating-point order.
- Added a cooperative species terminal. Species lanes share velocity data,
  aggregate ambient state, and publish gas, ambient, velocity, and pressure in
  one dispatch.
- Retained canonical fallbacks for unsupported species counts and non-formal
  frames.
- Added multi-frame exact tests for pressure, density, cooperative terminal,
  partial dimensions, and fallback paths.

## Heat

- Added four-cell ambient exchange/feedback aggregation and fused it into the
  bridge-aware cell-heat path.
- Fused condense gas application for the fixed four-by-six cell/species layout.
- Added `terminal4x6`, combining cell targets, condensation, six gas layers,
  bridge publication, aux updates, and dirty-tile output.
- Added workgroup dirty aggregation to reduce per-cell global atomics.
- Added sparse bridge-aux residency under a strict formal-frame gate. Fallback
  frames rehydrate private state before use.
- Added in-place sparse terminal writes after a workgroup barrier guarantees all
  sampled reads precede writes.
- Reused split-target active state and elided dead condensation stores.
- Added lazy action inputs so timer, temperature, integrity, velocity, and
  displaced material are fetched only for reachable actions.
- Packed phase and boil targets into one word with table-generation guards and
  canonical fallback when tables change incompatibly.
- Added hierarchical row summaries and a runtime-probed NV32 ballot reduction
  for terminal gas/empty counts.
- Fused terminal cell-aux dirty publication and direct ambient bridge output.
- Added raw-state tests for terminal variants, lazy inputs, packed targets,
  sparse residency, row summaries, NV32 ballot, condensation, and partial grids.

## Liquid

- Added an NV32 tile solver using packed shared rows, warp ballots, shuffles, and
  direct vertical segment mapping.
- Replaced shared atomic change tracking with a per-lane flag and one warp vote
  per tile.
- Added provenance row streaming so tile and seam passes carry compact source
  identity and the terminal pass materializes authoritative cell state once.
- Fused tile-solve bridge hydration and snapshot output.
- Compacted the tile snapshot, elided duplicate state, and packed immutable
  pre-state separately from mutable blocker bits.
- Added lazy placeholder role evaluation.
- Added row-leader seam-X processing and four-row workgroup packing.
- Added seam-Y shared snapshots to preserve pre-write boundary state without
  redundant global reads.
- Skipped seam-prefetch dispatches when a fully active grid already has all
  required data resident.
- Fused sink and float buoyancy passes and reused a shared vertical sink halo.
- Added buoyancy cleanup split, snapshot pre-state, shared sink cache, and narrow
  blocker/displaced hydration under strict formal gates.
- Added a 16x16 cleanup specialization and fused bridge aux cleanup.
- Added flow-intent shared forward halos, shared provenance metadata, and lazy
  aux reads for cells that survive packed-state rejection.
- Added provenance initialization fusion for full-active formal frames.
- Retained indirect active-tile dispatch and authoritative scheduler refresh;
  this changes dispatched work, not the simulated active region.
- Added exact tests for tile conflicts, lane voting, snapshots, seams, buoyancy,
  cleanup, provenance, partial grids, and fallback gates.

## Reactions

- Added expanded active-tile masks and reused GPU-authoritative scheduler state
  instead of repeatedly uploading dense masks.
- Added separate cell-core and velocity roles so passes that do not modify
  velocity do not copy it across the full grid.
- Added rule/action plan caches keyed by rule-table generations.
- Added authoritative per-material candidate masks and packed rule descriptors
  to avoid testing impossible rules for every cell.
- Added a guarded fused material-material/material-gas/material-light pass.
  Unsupported actions, RHS consumption, deferred outputs, wildcard cases, and
  malformed descriptors fall back to canonical passes.
- Added a 32x8 row-oriented terminal specialization, local16 fallback, shared
  transpose support, and fast equal-state publication.
- Published the material-triplet terminal directly to motion and integrated the
  same-frame velocity result when the strict handoff gate is satisfied.
- Added packed timed and self emit-target worklists.
- Added timed emit-target producer output while action data is resident.
- Added self gas-candidate worklists and authoritative LHS masks.
- Added material spans and direct action spans for self rules, retaining fallback
  for unsupported rule shapes.
- Added timed/self authoritative segment masks and packed cell-flag metadata.
- Added lazy segment-metadata zeroing and fused light-counter initialization.
- Replaced full flow-source payload clears with generation validity and an
  8-bit token; token wrap performs a physical clear.
- Fused terminal gas-delta application with bridge publication.
- Added tests for rule-plan invalidation, descriptors, action fallbacks, flow
  generations, timed/self metadata, emit worklists, gas candidates, terminal
  handoff, and gas publication.

## Motion

- Added a private compact powder-reservation ABI and lazy expansion to the public
  ABI for debug, serialization, readback, and CPU ownership transitions.
- Replaced powder apply-index clears with generation epochs.
- Added local64 target clear and fused aux index scratch data.
- Precomputed fallback blockers during reservation generation.
- Classified terminal trivial-blocked reservations before resolve.
- Added provisional-moving and nontrivial-resolve worklists so blocked records do
  not execute the full conflict resolver.
- Deduplicated apply-tile publication within each workgroup before global atomics.
- Added source-indexed direct apply for strictly gated GPU-generated disjoint
  reservations; uploaded/public reservations retain the canonical path.
- Added sparse powder bridge publication for touched source/target cells.
- Added minimal bridge hydration for falling-island materialization and resolve.
- Avoided rewriting unchanged falling-island materialization/apply cells.
- Fused falling-island materialization/apply with bridge publication.
- Added reaction-latch clear to the terminal handoff and collapse dirty-queue
  generation to final integration.
- Added exact tests for compact reservations, worklists, blocker fallback,
  trivial classification, tile deduplication, changed-only island paths,
  minimal hydration, and handoff output.

## Optics

- Added direct bridge-visible publication and direct visual-accumulator compose.
- Added trace-seeded sparse cell, gas, and visible worklists with indirect
  dispatch.
- Added generation-marked sparse tile lists and previous-frame sparse clearing
  instead of full-grid clears.
- Fused gas-visible expansion into worklist construction.
- Added runtime-probed NV32 gas-owner compaction. A warp reserves one contiguous
  output range instead of one global atomic per entry.
- Added a bounded trace-stack specialization when rule metadata proves the stack
  limit; general rules retain the canonical stack.
- Elided full-active mask hydration only when authoritative TTL state proves the
  entire grid active.
- Fused reaction-latch clear into sparse optics termination.
- Added multi-frame exact tests for sparse worklists, tile generation, bounded
  stacks, full-active/partial fallback, and NV32 owner compaction.

## Disabled Or Fallback Implementations

The commit retains several compiled fallback or experimental implementations so
the validated production path can be compared or used on unsupported hardware.
They are not included in the performance claim:

- Collapse support tile-union and atomic-union, support image-barrier elision,
  support/outcome publish fusion, and packed incremental snapshot.
- Gas pressure dependency-cone projection and force/thermo fusion.
- Heat diffuse4 cell fusion, terminal 16x8, broad aux/gas residency, phase fusion,
  and deferred dirty handoff.
- Liquid no-liquid fast path, broad aux residency, flow bridge residency, active
  mask cache, cleanup/flow fusion, active-decay fusion, liquid-kind cache, and
  provenance cleanup-terminal fusion.
- Reaction timed/self same-dispatch, timed/self sparse in-place, cached dense self
  state, and fused self gas output.
- Motion broad direct bridge apply and generated-path reuse.
- Optics tile-local atomic-max worklist construction.

## Verification

- Two final 512-frame `1920x1080 random_materials` benchmarks.
- Raw-byte exact checkers for retained collapse generations, JFA, materialize,
  liquid provenance, gas pressure, heat terminal, reaction generations and
  descriptors, motion reservations, and optics sparse state.
- Shader template verification: 371 shaders checked, zero failures.
- Python compilation and `git diff --check`.
- All split regression modules outside the legacy monolithic engine test:
  `294 passed in 23.99s`.
- A full-suite run before the final four test-fixture alignments reported
  `1607 passed, 186 failed`. Of those failures, 182 were concentrated in
  `tests/test_engine_core.py`; the other four stale source-layout or public-ABI
  assertions were corrected and are included in the 294 passing modular tests.
  The legacy engine-core failures were retained rather than deleted or broadly
  rewritten to manufacture a green result.
- The old label-union mock signature and pre-v3 phase-0 round-count assertions
  were updated to the current production interfaces before this commit.

Generated benchmark artifacts are intentionally not committed. Re-run
`scripts/benchmark_engine.py` and write output under `./tmp/` when new evidence
is required.
