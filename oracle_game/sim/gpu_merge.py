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

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

LOCAL_SIZE = 8


@dataclass(slots=True)
class MergeCandidates:
    heat: dict[str, Any]
    reactions: dict[str, Any]
    motion: dict[str, Any]
    liquid: dict[str, Any]


class GPUMergePipeline:
    def __init__(self) -> None:
        self.programs: dict[str, Any] = {}
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}}

    def available(self, world: "WorldEngine") -> bool:  # type: ignore[name-defined]
        if getattr(world, "simulation_backend", "gpu") == "cpu":
            return False
        bridge = world.bridge
        return bool(bridge.enabled and bridge.ctx is not None and bridge.ctx.version_code >= 430)

    def reset_pass_profile(self) -> None:
        self.last_pass_profile = {"passes": [], "summary": {}}

    @contextmanager
    def _profile_pass(self, world: "WorldEngine", name: str):  # type: ignore[name-defined]
        profile = self.last_pass_profile if bool(getattr(world, "profile_passes_enabled", False)) else None
        ctx = world.bridge.ctx if bool(getattr(world, "profile_passes_sync", False)) else None
        if profile is not None and ctx is not None:
            ctx.finish()
        start = time.perf_counter() if profile is not None else 0.0
        try:
            yield
        finally:
            if profile is None:
                return
            if ctx is not None:
                ctx.finish()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            profile["passes"].append({"name": str(name), "cpu_ms": elapsed_ms, "gpu_ms": elapsed_ms if ctx is not None else None})
            summary = profile["summary"].setdefault(str(name), {"count": 0, "cpu_ms": 0.0, "gpu_ms": None})
            summary["count"] += 1
            summary["cpu_ms"] += elapsed_ms
            if ctx is not None:
                summary["gpu_ms"] = float(summary["gpu_ms"] or 0.0) + elapsed_ms

    def _ensure_programs(self, ctx: Any) -> None:
        if "merge_cell_core" in self.programs:
            return
        self.programs["merge_cell_core"] = ctx.compute_shader(
            f"""
            #version 430
            layout(local_size_x={LOCAL_SIZE}, local_size_y={LOCAL_SIZE}, local_size_z=1) in;
            uniform ivec2 cell_grid_size;

            layout(std430, binding=0) buffer BridgeCellCoreBuffer {{
                uint cell_core[];
            }};

            // heat candidate
            layout(binding=0) uniform sampler2D heat_material_tex;
            layout(binding=1) uniform sampler2D heat_phase_tex;
            layout(binding=2) uniform sampler2D heat_temp_tex;
            layout(binding=3) uniform sampler2D heat_integrity_tex;
            layout(binding=4) uniform sampler2D heat_flags_tex;
            // reactions candidate
            layout(binding=5) uniform sampler2D rxn_material_tex;
            layout(binding=6) uniform sampler2D rxn_phase_tex;
            layout(binding=7) uniform sampler2D rxn_temp_tex;
            layout(binding=8) uniform sampler2D rxn_integrity_tex;
            layout(binding=9) uniform sampler2D rxn_timer_tex;
            layout(binding=10) uniform sampler2D rxn_latched_tex;
            layout(binding=11) uniform sampler2D rxn_velocity_tex;
            // motion candidate
            layout(binding=12) uniform sampler2D motion_material_tex;
            layout(binding=13) uniform sampler2D motion_phase_tex;
            layout(binding=14) uniform sampler2D motion_velocity_tex;
            // liquid candidate
            layout(binding=15) uniform sampler2D liquid_material_tex;
            layout(binding=16) uniform sampler2D liquid_phase_tex;
            layout(binding=17) uniform sampler2D liquid_flags_tex;
            layout(binding=18) uniform sampler2D liquid_timer_tex;
            layout(binding=19) uniform sampler2D liquid_velocity_tex;

            layout(r32f, binding=0) writeonly uniform image2D bridge_material_img;

            uint pack_timer(vec4 timer) {{
                uvec4 value = uvec4(clamp(round(timer), vec4(0.0), vec4(255.0)));
                return value.x | (value.y << 8u) | (value.z << 16u) | (value.w << 24u);
            }}

            bool changed_f(float a, float b) {{
                return a != b;
            }}
            bool changed_v(vec2 a, vec2 b) {{
                return a.x != b.x || a.y != b.y;
            }}

            void main() {{
                ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                if (gid.x >= cell_grid_size.x || gid.y >= cell_grid_size.y) {{
                    return;
                }}
                int idx = (gid.y * cell_grid_size.x + gid.x) * 5;
                uint w0 = cell_core[idx];
                uint w1 = cell_core[idx + 1];
                uint w2 = cell_core[idx + 2];
                uint w3 = cell_core[idx + 3];
                uint w4 = cell_core[idx + 4];

                float prev_material = float(w0 & 0xFFFFu);
                float prev_phase = float((w0 >> 16) & 0xFFu);
                float prev_flags = float((w0 >> 24) & 0xFFu);
                vec2 prev_velocity = unpackHalf2x16(w1);
                float prev_temp = uintBitsToFloat(w2);
                vec4 prev_timer = vec4(
                    float(w3 & 0xFFu),
                    float((w3 >> 8) & 0xFFu),
                    float((w3 >> 16) & 0xFFu),
                    float((w3 >> 24) & 0xFFu)
                );
                float prev_integrity = float(w4 & 0xFFFFu);

                float heat_material = texelFetch(heat_material_tex, gid, 0).x;
                float heat_phase = texelFetch(heat_phase_tex, gid, 0).x;
                float heat_temp = texelFetch(heat_temp_tex, gid, 0).x;
                float heat_integrity = texelFetch(heat_integrity_tex, gid, 0).x;
                float heat_flags = texelFetch(heat_flags_tex, gid, 0).x;

                float rxn_material = texelFetch(rxn_material_tex, gid, 0).x;
                float rxn_phase = texelFetch(rxn_phase_tex, gid, 0).x;
                float rxn_temp = texelFetch(rxn_temp_tex, gid, 0).x;
                float rxn_integrity = texelFetch(rxn_integrity_tex, gid, 0).x;
                vec4 rxn_timer = texelFetch(rxn_timer_tex, gid, 0);
                float rxn_latched = texelFetch(rxn_latched_tex, gid, 0).x;
                vec2 rxn_velocity = texelFetch(rxn_velocity_tex, gid, 0).xy;

                float motion_material = texelFetch(motion_material_tex, gid, 0).x;
                float motion_phase = texelFetch(motion_phase_tex, gid, 0).x;
                vec2 motion_velocity = texelFetch(motion_velocity_tex, gid, 0).xy;

                float liquid_material = texelFetch(liquid_material_tex, gid, 0).x;
                float liquid_phase = texelFetch(liquid_phase_tex, gid, 0).x;
                float liquid_flags = texelFetch(liquid_flags_tex, gid, 0).x;
                vec4 liquid_timer = texelFetch(liquid_timer_tex, gid, 0);
                vec2 liquid_velocity = texelFetch(liquid_velocity_tex, gid, 0).xy;

                // material: reactions > motion > liquid > heat
                float material = prev_material;
                if (changed_f(rxn_material, prev_material)) material = rxn_material;
                else if (changed_f(motion_material, prev_material)) material = motion_material;
                else if (changed_f(liquid_material, prev_material)) material = liquid_material;
                else if (changed_f(heat_material, prev_material)) material = heat_material;

                // phase: reactions > motion > liquid > heat
                float phase = prev_phase;
                if (changed_f(rxn_phase, prev_phase)) phase = rxn_phase;
                else if (changed_f(motion_phase, prev_phase)) phase = motion_phase;
                else if (changed_f(liquid_phase, prev_phase)) phase = liquid_phase;
                else if (changed_f(heat_phase, prev_phase)) phase = heat_phase;

                // temp: additive heat + (reactions - prev)
                float temp = heat_temp + (rxn_temp - prev_temp);

                // integrity: heat > reactions
                float integrity = prev_integrity;
                if (changed_f(heat_integrity, prev_integrity)) integrity = heat_integrity;
                else if (changed_f(rxn_integrity, prev_integrity)) integrity = rxn_integrity;

                // timer: reactions (falls through to prev when reactions unchanged)
                vec4 timer = prev_timer;
                if (rxn_timer != prev_timer) timer = rxn_timer;
                else if (liquid_timer != prev_timer) timer = liquid_timer;

                // flags: liquid/heat candidates carry the full flags byte (prev | their
                // changes); reactions only sets the REACTION_LATCHED bit (bit 1 == 2).
                float flags;
                if (changed_f(liquid_flags, prev_flags)) flags = liquid_flags;
                else if (changed_f(heat_flags, prev_flags)) flags = heat_flags;
                else flags = prev_flags;
                if (rxn_latched != 0.0) {{
                    flags = float(uint(flags + 0.5) | 2u);
                }}

                // velocity: motion > liquid
                vec2 velocity = prev_velocity;
                if (changed_v(motion_velocity, prev_velocity)) velocity = motion_velocity;
                else if (changed_v(liquid_velocity, prev_velocity)) velocity = liquid_velocity;
                else if (changed_v(rxn_velocity, prev_velocity)) velocity = rxn_velocity;

                uint material_id = uint(material + 0.5) & 0xFFFFu;
                uint phase_id = uint(phase + 0.5) & 0xFFu;
                uint flags_id = uint(flags + 0.5) & 0xFFu;
                cell_core[idx] = material_id | (phase_id << 16u) | (flags_id << 24u);
                cell_core[idx + 1] = packHalf2x16(velocity);
                cell_core[idx + 2] = floatBitsToUint(temp);
                cell_core[idx + 3] = pack_timer(timer);
                cell_core[idx + 4] = uint(integrity + 0.5) & 0xFFFFu;

                imageStore(bridge_material_img, gid, vec4(material, 0.0, 0.0, 0.0));
            }}
            """
        )

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
