"""Shared base class for GPU compute pipelines.

The simulation ships one ``GPUxPipeline`` class per update stage (heat, gas,
liquid, motion, collapse, reactions, optics, …) plus a few frame-level
pipelines (merge, placeholders, page stripes, world commands).  Historically
each pipeline copy-pasted the same handful of small helpers.  This module owns
those helpers once so the per-stage files only contain stage-specific logic.

Behavior note: the helpers here are byte-for-byte / semantically identical to
the bodies that were inlined in every pipeline (verified during the structural
refactor of 2026-07).  Subclassing this base must not change any observable
behavior — it only removes duplication.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a circular import at runtime; WorldEngine is passed in
    from oracle_game.world import WorldEngine


class GPUPipelineBase:
    """Common lifecycle/profiling helpers shared by every GPU pipeline.

    Subclasses provide stage-specific resource building, program compilation
    and the dispatch entry point (``step`` / ``apply`` / ``merge_cell_core`` …).
    Override :meth:`_barrier_bits` to extend the memory-barrier mask used by
    :meth:`_sync_compute_writes` (e.g. pipelines using indirect dispatch add
    ``COMMAND_BARRIER_BIT``).
    """

    # --- profile / availability ---------------------------------------------

    def available(self, world: "WorldEngine") -> bool:
        """Whether the GPU path may run for this world.

        Mirrors the check every pipeline inlined: the CPU backend opts out, and
        we require a live GL 4.3+ context on the bridge.
        """
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        bridge = world.bridge
        return bool(bridge.enabled and bridge.ctx is not None and bridge.ctx.version_code >= 430)

    def reset_pass_profile(self) -> None:
        """Clear accumulated per-pass profile data."""
        self.last_pass_profile = {"passes": [], "summary": {}}

    @contextmanager
    def _profile_pass(self, world: "WorldEngine", name: str):
        """Time one GPU pass when ``world.profile_passes_enabled`` is set.

        When profiling is disabled this is a bare ``yield`` (no ctx.finish()
        stalling), matching the original inlined bodies.  When enabled and
        ``profile_passes_sync`` is set we force a GL finish so the timer
        captures GPU time rather than just CPU submit time.
        """
        profile = self.last_pass_profile if bool(getattr(world, "profile_passes_enabled", False)) else None
        ctx = world.bridge.ctx if bool(getattr(world, "profile_passes_sync", False)) else None
        if profile is not None and ctx is not None:
            ctx.finish()
        start = time.perf_counter() if profile is not None else 0.0
        try:
            yield
        finally:
            # NB: do NOT `return` here when profile is None — a `return` inside
            # `finally` would SWALLOW any exception raised in the `with` body
            # (e.g. _require_gpu_stage RuntimeError). Only record timing when
            # profiling is enabled; otherwise let exceptions propagate.
            if profile is not None:
                if ctx is not None:
                    ctx.finish()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                entry = {
                    "name": str(name),
                    "cpu_ms": elapsed_ms,
                    "gpu_ms": elapsed_ms if ctx is not None else None,
                }
                profile["passes"].append(entry)
                summary = profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
                summary["count"] += 1
                summary["cpu_ms"] += elapsed_ms
                if ctx is not None:
                    summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

    # --- backend / uniform helpers ------------------------------------------

    def _formal_gpu_frame(self, world: "WorldEngine") -> bool:
        """True only during an authoritative GPU simulation frame.

        Guards GPU-authoritative writes so that preview / CPU-oracle paths do
        not touch the bridge's authoritative resources.
        """
        return (
            getattr(world, "simulation_backend", "") == "gpu"
            and bool(getattr(world, "_world_simulation_frame_active", False))
        )

    def _set_uniform_if_present(self, program: Any, name: str, value: Any) -> None:
        """Set a uniform, silently ignoring programs that don't declare it."""
        try:
            program[name].value = value
        except KeyError:
            return

    def _barrier_bits(self) -> tuple[str, ...]:
        """Names of ``ctx`` memory-barrier bits to OR together.

        Default covers image writes + texture fetch + shader storage — the
        bits every compute pass needs.  Pipelines using indirect dispatch or
        framebuffer interop override this to add ``COMMAND_BARRIER_BIT`` /
        ``FRAMEBUFFER_BARRIER_BIT`` / ``BUFFER_UPDATE_BARRIER_BIT``.
        """
        return (
            "SHADER_IMAGE_ACCESS_BARRIER_BIT",
            "TEXTURE_FETCH_BARRIER_BIT",
            "SHADER_STORAGE_BARRIER_BIT",
        )

    def _sync_compute_writes(self, ctx: Any) -> None:
        """Issue a ``memory_barrier`` over :meth:`_barrier_bits`.

        Uses ``getattr`` with a 0 fallback so the same code is safe across GL
        versions where a given bit constant may be absent.
        """
        bits = 0
        for name in self._barrier_bits():
            bits |= getattr(ctx, name, 0)
        ctx.memory_barrier(bits)
