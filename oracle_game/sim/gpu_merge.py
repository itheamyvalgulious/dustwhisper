"""Phase C fixed-priority merge pass.

Reads the candidate cell-state textures from heat / reactions / motion / liquid
(each system's publish-source scratch, left un-published this frame) plus the
bridge cell_core (which still holds last frame's final state because no system
published mid-frame) and writes a single merged cell_core.

Field rules (plan/step2.md "新数据流" -> "合并规则第一版"):
  material / phase / timer / flags : priority reactions > motion > liquid > heat;
                                      a candidate "changes" a cell when its value
                                      differs from prev (last frame). Unchanged
                                      candidates fall through to prev.
  temp                             : additive  heat_temp + (reactions_temp - prev_temp)
                                      (heat conduction + reaction exotherm; motion
                                      and liquid do not modify cell temperature).
  integrity                        : priority heat > reactions.
  velocity                         : priority motion > liquid.

island_id / entity_id / placeholder_displaced_material are taken from motion then
liquid (motion writes island settlements; liquid writes placeholder displacement).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_game.sim.gpu_base import GPUPipelineBase
from oracle_game.sim.shader_loader import build_compute_shader

LOCAL_SIZE = 8


@dataclass(slots=True)
class MergeCandidates:
    heat: dict[str, Any]
    reactions: dict[str, Any]
    motion: dict[str, Any]
    liquid: dict[str, Any]


class GPUMergePipeline(GPUPipelineBase):
    # ``available`` / ``reset_pass_profile`` / ``_profile_pass`` are inherited
    # from :class:`GPUPipelineBase` (formerly copy-pasted here verbatim).

    def __init__(self) -> None:
        self.programs: dict[str, Any] = {}
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    def _ensure_programs(self, ctx: Any) -> None:
        if "merge_cell_core" in self.programs:
            return
        self.programs["merge_cell_core"] = build_compute_shader(ctx, "merge/merge_cell_core.comp", {"LOCAL_SIZE": LOCAL_SIZE})

    def merge_cell_core(
        self,
        world: "WorldEngine",  # type: ignore[name-defined]
        candidates: MergeCandidates,
    ) -> bool:
        if not self.available(world):
            return False
        ctx = world.bridge.ctx
        if ctx is None:
            return False
        bridge = world.bridge
        bridge.ensure_world_resources(world)
        self._ensure_programs(ctx)
        program = self.programs["merge_cell_core"]
        program["cell_grid_size"].value = (world.width, world.height)
        h = candidates.heat
        r = candidates.reactions
        m = candidates.motion
        l = candidates.liquid
        h["material"].use(location=0)
        h["phase"].use(location=1)
        h["temp"].use(location=2)
        h["integrity"].use(location=3)
        h["flags"].use(location=4)
        r["material"].use(location=5)
        r["phase"].use(location=6)
        r["temp"].use(location=7)
        r["integrity"].use(location=8)
        r["timer"].use(location=9)
        r["latched"].use(location=10)
        r["velocity"].use(location=11)
        m["material"].use(location=12)
        m["phase"].use(location=13)
        m["velocity"].use(location=14)
        l["material"].use(location=15)
        l["phase"].use(location=16)
        l["flags"].use(location=17)
        l["timer"].use(location=18)
        l["velocity"].use(location=19)
        bridge.buffers["cell_core"].bind_to_storage_buffer(binding=0)
        bridge.textures["material"].bind_to_image(0, read=False, write=True)
        group_x = (world.width + LOCAL_SIZE - 1) // LOCAL_SIZE
        group_y = (world.height + LOCAL_SIZE - 1) // LOCAL_SIZE
        with self._profile_pass(world, "merge_cell_core"):
            program.run(group_x, group_y, 1)
            ctx.memory_barrier(
                ctx.SHADER_IMAGE_ACCESS_BARRIER_BIT
                | ctx.SHADER_STORAGE_BARRIER_BIT
                | ctx.TEXTURE_FETCH_BARRIER_BIT
            )
        bridge.mark_gpu_authoritative("cell_core", "material")
        return True

    def release(self) -> None:
        for program in self.programs.values():
            try:
                program.release()
            except Exception:
                pass
        self.programs.clear()
