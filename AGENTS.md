# Project Instructions

- Do not reduce simulation density, resolution, active region size, solver coverage, realtime budget thresholds, visual fidelity, or configured behavior to claim a performance win.
- `enginedemo` performance target: 1920x1080 viewport with full active random materials must sustain 60 FPS on the GPU path.
- Performance work must start from pass-level profiling evidence and optimize the measured hot passes directly. Do not skip solver stages or lower configuration as a substitute for optimization.
- Do not write benchmark artifacts, timing outputs, or temporary work files to `/tmp`. Use the project-local `./tmp/` directory for temporary output.

# Tooling

- Lint/format: `ruff check` and `ruff format` (config in `pyproject.toml`, line-length 100, rules F+I). Keep them clean.
- Tests: `pytest` (config in `pyproject.toml`). GPU-dependent tests should use the `gl_context`/`require_gpu` fixtures from `tests/conftest.py` so they skip cleanly on machines without a GPU.
- Shader integrity: `python scripts/verify_shaders.py` validates every `oracle_game/shaders/<stage>/*.comp` marker against the owning stage's substitutions; run it after touching shaders or `_SHADER_SUBS`.
- Behavior gate: `python scripts/behavior_snapshot.py` must stay byte-identical across simulation changes (record the hash before and after).
- `tests/test_engine_core.py` can crash (SIGSEGV) in long single-process runs; split large runs into chunks.

# Known issues

- NVIDIA EGL surfaceless contexts (observed on driver 580.159.03 / GTX 1650) are instance-level unreliable for compute work: a few percent of freshly created contexts mis-read uniforms / SSBO content in compute dispatches, while `glCopyBufferSubData` and buffer map reads stay coherent. No GL-level cure was found (memory barriers including `CLIENT_MAPPED_BUFFER_BARRIER_BIT`, `glFinish`, orphaning, rebinding, buffer pre-allocation were all ruled out empirically). Consequences:
  - Buffer-backed GPU readback windows (cell_core / gas / plain buffers) are filled via DMA copies (`glCopyBufferSubData` → map → slot write) instead of compute pack shaders; texture windows still use shaders.
  - Simulations run through the HTTP console (own context per console) can still silently lose or mis-read GPU state on a bad context instance; tests that assert console-tick readback contents are inherently flaky on this driver. Split `tests/test_engine_core.py` runs into chunks and treat rare content mismatches in the non-symmetric paging console tests accordingly.
