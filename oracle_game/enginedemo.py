from __future__ import annotations

from dataclasses import replace
import threading
import time as _time
from typing import Any

import numpy as np

from oracle_game.http_console import EngineHTTPConsole, EngineRunState
from oracle_game.types import DebugView
from oracle_game.world import WorldEngine

from oracle_game.demo_input import (
    BRUSH_MODES,
    BRUSH_MODE_CYCLE_KEYS,
    CONTROLLER_TOGGLE_KEYS,
    DEBUG_VIEW_KEYMAP,
    GAS_SPECIES_CYCLE_KEYS,
    MATERIAL_KEYS,
    OPTICS_LIGHT_CYCLE_KEYS,
    RESET_WORLD_KEYS,
    _alpha_keys,
    apply_demo_paint,
    clamp_demo_brush_radius,
    cycle_demo_brush_mode,
    cycle_demo_named_choice,
    demo_debug_view_for_key,
    demo_force_direction_from_drag,
    demo_light_direction_and_spread_from_drag,
    demo_material_for_key,
    demo_screen_to_buffer_cell,
    demo_screen_to_world_cell,
    demo_velocity_from_drag,
    is_demo_brush_cycle_key,
    is_demo_controller_toggle_key,
    is_demo_gas_cycle_key,
    is_demo_optics_cycle_key,
    is_demo_reset_key,
    queue_demo_paint,
    resolve_demo_paint_command,
)
from oracle_game.demo_render import (
    DEMO_AMBIENT_BOTTOM_LIGHT,
    DEMO_AMBIENT_TOP_LIGHT,
    DEMO_PATTERN_SCALE,
    apply_demo_render_uniforms,
    build_demo_render_uniforms,
)
from oracle_game.demo_sizing import (
    DEMO_ACTIVE_SCALE,
    DEMO_CONTROLLER_ENTITY_ID,
    DEMO_LOGICAL_WORLD_SCALE,
    DEMO_TARGET_CELL_PIXELS,
    ENGINE_DEMO_TITLE,
    _demo_shadow_material_name,
    _format_demo_probe_rgb,
    _format_demo_probe_vector,
    build_demo_controller_entities,
    build_demo_controller_entities_for_world_focus,
    build_demo_controller_probe_entity,
    build_demo_controller_state,
    compute_demo_grid_sizing,
    demo_backend_report,
    demo_brush_selection_label,
    demo_default_focus_world,
    demo_display_material_name,
    demo_view_focus_label,
    format_demo_controller_observation_summary,
    format_demo_controller_status,
    format_demo_focus_probe,
    format_demo_status_title,
    request_demo_redraw,
)


DEMO_REALTIME_BUDGET_CELL_THRESHOLD = 1_000_000
DEMO_DEBUG_TEXTURE_REFRESH_SECONDS = 1.0 / 15.0
DEMO_VERTEX_SHADER_SOURCE = """
    #version 330
    in vec2 in_pos;
    in vec2 in_uv;
    out vec2 v_uv;
    void main() {
        v_uv = in_uv;
        gl_Position = vec4(in_pos, 0.0, 1.0);
    }
"""
DEMO_FRAGMENT_SHADER_SOURCE = """
    #version 330
    uniform sampler2D material_tex;
    uniform sampler2D light_tex;
    uniform sampler2D debug_tex;
    uniform sampler2D atlas_tex;
    uniform ivec2 buffer_size;
    uniform ivec2 active_size;
    uniform ivec2 buffer_origin;
    uniform ivec2 world_origin;
    uniform ivec2 atlas_grid;
    uniform int view_mode;
    uniform bool force_debug_texture;
    uniform float pattern_scale;
    uniform vec3 ambient_top_light;
    uniform vec3 ambient_bottom_light;
    in vec2 v_uv;
    out vec4 fragColor;
    void main() {
        ivec2 raw_display_cell = ivec2(clamp(floor(v_uv * vec2(active_size)), vec2(0.0), vec2(active_size) - 1.0));
        ivec2 display_cell = ivec2(raw_display_cell.x, active_size.y - 1 - raw_display_cell.y);
        ivec2 cell = ivec2(
            (display_cell.x + buffer_origin.x) % buffer_size.x,
            (display_cell.y + buffer_origin.y) % buffer_size.y
        );
        ivec2 logical_cell = world_origin + display_cell;
        vec3 light_rgb = clamp(texelFetch(light_tex, cell, 0).rgb, 0.0, 2.0);
        float top_factor = 1.0 - float(display_cell.y) / max(1.0, float(active_size.y - 1));
        vec3 ambient_light = mix(ambient_bottom_light, ambient_top_light, top_factor);
        if (force_debug_texture || view_mode != 0) {
            fragColor = vec4(texelFetch(debug_tex, cell, 0).rgb, 1.0);
            return;
        }
        float material_id = texelFetch(material_tex, cell, 0).r;
        if (material_id < 0.5) {
            vec3 sky = mix(vec3(0.08, 0.10, 0.13), vec3(0.16, 0.20, 0.27), top_factor);
            fragColor = vec4(clamp(sky + light_rgb * 0.45, 0.0, 1.0), 1.0);
            return;
        }
        int mid = int(material_id + 0.5);
        int atlas_x = mid % atlas_grid.x;
        int atlas_y = mid / atlas_grid.x;
        vec2 repeat_uv = fract(vec2(logical_cell) / pattern_scale);
        vec2 atlas_uv = (vec2(atlas_x, atlas_y) + repeat_uv) / vec2(atlas_grid);
        vec3 base = texture(atlas_tex, atlas_uv).rgb;
        vec3 color = base * (ambient_light + clamp(light_rgb, 0.0, 1.5)) + light_rgb * 0.65;
        fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
    }
"""


def main() -> None:
    try:
        import moderngl  # noqa: F401
        import moderngl_window as mglw
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install moderngl and moderngl-window inside the venv before running enginedemo.") from exc

    class EngineDemo(mglw.WindowConfig):
        gl_version = (4, 3)
        title = ENGINE_DEMO_TITLE
        window_size = (1440, 900)
        vsync = False
        resizable = True
        aspect_ratio = None

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            sizing = compute_demo_grid_sizing(self.wnd.width, self.wnd.height)
            self.demo_visible_width = sizing["visible_width"]
            self.demo_visible_height = sizing["visible_height"]
            self.demo_logical_world_width = sizing["logical_world_width"]
            self.demo_logical_world_height = sizing["logical_world_height"]
            self.engine = WorldEngine(
                width=sizing["buffer_width"],
                height=sizing["buffer_height"],
                active_width=sizing["active_width"],
                active_height=sizing["active_height"],
                gpu_context=self.ctx,
            )
            self.engine.demo_visible_width = self.demo_visible_width
            self.engine.demo_visible_height = self.demo_visible_height
            self.engine.gpu_realtime_budget_enabled = True
            self.engine.gpu_realtime_budget_cell_threshold = DEMO_REALTIME_BUDGET_CELL_THRESHOLD
            self._bind_gui_gpu_context()
            prewarm_collapse = getattr(self.engine, "prewarm_formal_connected_collapse", None)
            if callable(prewarm_collapse):
                prewarm_collapse()
            self.state = EngineRunState()
            try:
                self.http = EngineHTTPConsole(self.engine, self.state, own_gpu_context=False)
            except TypeError as exc:
                if "own_gpu_context" not in str(exc):
                    raise
                self.http = EngineHTTPConsole(self.engine, self.state)
                setattr(self.http, "own_gpu_context", False)
            self.http.start()
            self.brush_radius = 3
            self.selected_material = MATERIAL_KEYS[0]
            self.brush_mode = BRUSH_MODES[0]
            self.debug_view = DebugView.MATERIAL
            self.gas_view_species = "water_gas"
            self.optics_view_light: str | None = None
            self.focus_x, self.focus_y = demo_default_focus_world(self.engine.paging)
            self.accumulator = 0.0
            self._last_present_time = _time.perf_counter()
            self._status_title = ""
            self.controller_debug_enabled = False
            self.controller_debug_cycle = 0
            self.controller_debug_dirty = False
            self.controller_debug_label: str | None = None
            self.controller_debug_saved_state: Any = None
            self.cpu_fps = 0.0
            self.gpu_fps = 0.0
            self.frame_ms = 0.0
            self.sim_ms = 0.0
            self.sync_ms = 0.0
            self.render_ms = 0.0
            self._cpu_fps_sample_time = _time.perf_counter()
            self._gpu_fps_sample_time = self._cpu_fps_sample_time
            self._cpu_frame_count = 0
            self._gpu_step_count = 0
            self._last_status_title_update_time = 0.0
            self._last_paint_key: tuple[object, ...] | None = None
            self._debug_frame_cache: np.ndarray | None = None
            self._debug_frame_cache_key: tuple[object, ...] | None = None
            self._last_debug_texture_upload_time = 0.0
            self._build_render_resources()
            self._prime_render_textures()
            self._refresh_status_title()

        def _bind_gui_gpu_context(self) -> None:
            bridge = self.engine.bridge
            if not hasattr(bridge, "ctx"):
                return
            if bridge.ctx is not self.ctx:
                bridge.attach_context(self.ctx)
                return
            bridge.enabled = True
            bridge.owner_thread_id = threading.get_ident()

        def _build_render_resources(self) -> None:
            self._bind_gui_gpu_context()
            self.program = self.ctx.program(
                vertex_shader=DEMO_VERTEX_SHADER_SOURCE,
                fragment_shader=DEMO_FRAGMENT_SHADER_SOURCE,
            )
            vertices = np.array(
                [
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype="f4",
            )
            self.vbo = self.ctx.buffer(vertices.tobytes())
            self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "in_pos", "in_uv")])
            with self.engine.state_lock:
                self.engine.bridge.ensure_world_resources(self.engine)
                self.program["material_tex"] = 0
                self.program["light_tex"] = 1
                self.program["debug_tex"] = 2
                self.program["atlas_tex"] = 3
                apply_demo_render_uniforms(
                    self.program,
                    build_demo_render_uniforms(self.engine, debug_view=self.debug_view),
                )

        def _prime_render_textures(self) -> None:
            with self.engine.state_lock:
                self._bind_gui_gpu_context()
                ensure_resources = getattr(self.engine.bridge, "ensure_world_resources", None)
                if callable(ensure_resources):
                    ensure_resources(self.engine)
                sync_display = getattr(self.engine.bridge, "sync_display_textures", None)
                if callable(sync_display):
                    sync_display(self.engine)
                    return
                sync_world = getattr(self.engine.bridge, "sync_world", None)
                if callable(sync_world):
                    try:
                        sync_world(self.engine, upload_debug_texture=False)
                    except TypeError as exc:
                        if "upload_debug_texture" not in str(exc):
                            raise
                        sync_world(self.engine)

        def on_render(self, time: float, frame_time: float) -> None:
            frame_start = _time.perf_counter()
            last_present_time = float(getattr(self, "_last_present_time", 0.0))
            if last_present_time > 0.0:
                target_frame_time = 1.0 / 60.0
                elapsed_since_present = frame_start - last_present_time
                sleep_time = target_frame_time - elapsed_since_present
                if sleep_time > 0.0:
                    _time.sleep(sleep_time)
                    frame_start = _time.perf_counter()
                frame_time = max(0.0, frame_start - last_present_time)
                self._last_present_time = frame_start
            bind_gpu_context = getattr(self, "_bind_gui_gpu_context", None)
            if callable(bind_gpu_context):
                bind_gpu_context()
            now = _time.perf_counter()
            self._record_cpu_frame(now)
            with self.engine.state_lock:
                stepped = False
                steps = 0
                sim_start = _time.perf_counter()
                if not self.state.paused:
                    self.accumulator += frame_time * max(0.1, self.state.speed)
                    self.accumulator = min(self.accumulator, 2.0 / 60.0)
                    while self.accumulator >= 1.0 / 60.0 and steps < 1:
                        self.engine.step(1.0 / 60.0)
                        self.accumulator -= 1.0 / 60.0
                        steps += 1
                        stepped = True
                elif self.state.single_step:
                    self.engine.step(1.0 / 60.0)
                    self.state.single_step = False
                    steps = 1
                    stepped = True
                sim_done = _time.perf_counter()
                self.sim_ms = (sim_done - sim_start) * 1000.0
                self._record_gpu_steps(steps, sim_done)

                if self.controller_debug_enabled and (self.controller_debug_dirty or stepped):
                    self._run_demo_controller_cycle(apply_turn=stepped)

                self.engine.default_debug_view = self.debug_view
                debug = None
                debug_refresh_due = False
                gpu_debug_synced = False
                gas_species_id = -1
                light_dose_channel = -1
                if self.debug_view != DebugView.MATERIAL:
                    if self.debug_view == DebugView.GAS:
                        gas_species_id = self.engine._resolve_sanctioned_gas_id(self.gas_view_species)
                    if self.debug_view in {DebugView.LIGHT, DebugView.OPTICS} and self.optics_view_light is not None:
                        light_id = self.engine._resolve_sanctioned_light_id(self.optics_view_light)
                        dose_channel = self.engine._shadow_light_dose_channel(light_id) if light_id >= 0 else None
                        light_dose_channel = -1 if dose_channel is None else int(dose_channel)
                sync_start = _time.perf_counter()
                if self.debug_view == DebugView.MATERIAL:
                    sync_display = getattr(self.engine.bridge, "sync_display_textures", None)
                    if callable(sync_display):
                        sync_display(self.engine)
                else:
                    gpu_backend = getattr(self.engine, "simulation_backend", "") == "gpu"
                    sync_debug_display = getattr(self.engine.bridge, "sync_debug_display_texture", None)
                    if callable(sync_debug_display):
                        gpu_debug_synced = bool(
                            sync_debug_display(
                                self.engine,
                                view=self.debug_view.value,
                                gas_species_id=gas_species_id,
                                light_dose_channel=light_dose_channel,
                            )
                        )
                    allow_cpu_debug_upload = not gpu_backend or not callable(sync_debug_display)
                    if not gpu_debug_synced and allow_cpu_debug_upload:
                        debug_key = (self.debug_view, self.gas_view_species, self.optics_view_light)
                        debug_refresh_due = (
                            getattr(self, "_debug_frame_cache", None) is None
                            or getattr(self, "_debug_frame_cache_key", None) != debug_key
                            or now - float(getattr(self, "_last_debug_texture_upload_time", 0.0))
                            >= DEMO_DEBUG_TEXTURE_REFRESH_SECONDS
                        )
                        if debug_refresh_due:
                            debug = self.engine.debug_frame(
                                self.debug_view,
                                gas_species=self.gas_view_species,
                                light_type=self.optics_view_light,
                            )
                            self._debug_frame_cache = debug
                            self._debug_frame_cache_key = debug_key
                            self._last_debug_texture_upload_time = now
                            try:
                                self.engine.bridge.sync_world(
                                    self.engine,
                                    debug_frame=debug,
                                    upload_debug_texture=True,
                                )
                            except TypeError as exc:
                                if "upload_debug_texture" not in str(exc):
                                    raise
                                self.engine.bridge.sync_world(self.engine, debug_frame=debug)
                sync_done = _time.perf_counter()
                self.sync_ms = (sync_done - sync_start) * 1000.0

                apply_demo_render_uniforms(
                    self.program,
                    build_demo_render_uniforms(self.engine, debug_view=self.debug_view),
                )
                self.engine.bridge.texture("material").use(0)
                self.engine.bridge.texture("light").use(1)
                self.engine.bridge.texture("debug").use(2)
                self.engine.bridge.atlas_texture().use(3)
                self.ctx.clear(0.02, 0.03, 0.05)
                self.vao.render(mode=self.ctx.TRIANGLE_STRIP)
                render_done = _time.perf_counter()
                self.render_ms = (render_done - sync_done) * 1000.0
                self.frame_ms = (render_done - frame_start) * 1000.0
                self.engine.demo_runtime_state = {
                    "frame_id": int(self.engine.frame_id),
                    "debug_view": self.debug_view.value,
                    "force_debug_texture": self.debug_view != DebugView.MATERIAL,
                    "visible_size": [int(self.demo_visible_width), int(self.demo_visible_height)],
                    "active_size": [int(self.engine.paging.active_width), int(self.engine.paging.active_height)],
                    "buffer_size": [int(self.engine.width), int(self.engine.height)],
                    "logical_world_size": [int(self.demo_logical_world_width), int(self.demo_logical_world_height)],
                    "origin": [int(self.engine.paging.origin_x), int(self.engine.paging.origin_y)],
                    "buffer_origin": [int(self.engine.paging.buffer_origin_x), int(self.engine.paging.buffer_origin_y)],
                    "cpu_fps": float(getattr(self, "cpu_fps", 0.0)),
                    "gpu_fps": float(getattr(self, "gpu_fps", 0.0)),
                    "frame_ms": float(self.frame_ms),
                    "sim_ms": float(self.sim_ms),
                    "sync_ms": float(self.sync_ms),
                    "render_ms": float(self.render_ms),
                    "backend_report": demo_backend_report(self.engine),
                }
                self._refresh_status_title()
                request_demo_redraw(self.wnd)

        def _record_cpu_frame(self, now: float) -> None:
            sample_time = float(getattr(self, "_cpu_fps_sample_time", now))
            frame_count = int(getattr(self, "_cpu_frame_count", 0)) + 1
            elapsed = max(0.0, float(now) - sample_time)
            if elapsed >= 0.5:
                self.cpu_fps = frame_count / elapsed
                self._cpu_frame_count = 0
                self._cpu_fps_sample_time = float(now)
                return
            self._cpu_frame_count = frame_count
            self._cpu_fps_sample_time = sample_time

        def _record_gpu_steps(self, steps: int, now: float) -> None:
            sample_time = float(getattr(self, "_gpu_fps_sample_time", now))
            step_count = int(getattr(self, "_gpu_step_count", 0)) + max(0, int(steps))
            elapsed = max(0.0, float(now) - sample_time)
            if elapsed >= 0.5:
                self.gpu_fps = step_count / elapsed
                self._gpu_step_count = 0
                self._gpu_fps_sample_time = float(now)
                return
            self._gpu_step_count = step_count
            self._gpu_fps_sample_time = sample_time

        def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
            self._paint_from_screen(x, y, dx=dx, dy=dy)

        def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
            if int(button) == 1:
                self._last_paint_key = None
                self._paint_from_screen(x, y, dx=0, dy=0)
                return
            self._focus_from_screen(x, y)

        def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
            with self.engine.state_lock:
                self.brush_radius = clamp_demo_brush_radius(self.brush_radius + int(y_offset))

        def on_key_event(self, key: int, action: int, modifiers: object) -> None:
            if action == 0:
                return
            with self.engine.state_lock:
                if (material := demo_material_for_key(key)) is not None:
                    self.selected_material = material
                elif (debug_view := demo_debug_view_for_key(key)) is not None:
                    self.debug_view = debug_view
                    self._debug_frame_cache = None
                    self._debug_frame_cache_key = None
                elif key == ord("["):
                    self.brush_radius = clamp_demo_brush_radius(self.brush_radius - 1)
                elif key == ord("]"):
                    self.brush_radius = clamp_demo_brush_radius(self.brush_radius + 1)
                elif key == ord("-"):
                    self.state.speed = max(0.1, self.state.speed * 0.8)
                elif key == ord("="):
                    self.state.speed = min(8.0, self.state.speed * 1.25)
                elif key == ord(" "):
                    self.state.paused = not self.state.paused
                elif key in (ord("N"), ord("n")):
                    self.state.single_step = True
                elif is_demo_optics_cycle_key(key):
                    self.optics_view_light = cycle_demo_named_choice(
                        self.optics_view_light,
                        (None, *self.engine.rulebook.lights_by_name.keys()),
                    )
                elif is_demo_gas_cycle_key(key):
                    gas_choices = tuple(name for name in self.engine.gas_name_by_id if name)
                    next_species = cycle_demo_named_choice(self.gas_view_species, gas_choices)
                    if next_species is not None:
                        self.gas_view_species = next_species
                elif is_demo_brush_cycle_key(key):
                    self.brush_mode = cycle_demo_brush_mode(self.brush_mode)
                elif is_demo_controller_toggle_key(key):
                    self._set_controller_debug_enabled(not self.controller_debug_enabled)
                elif is_demo_reset_key(key):
                    self.engine.reset_world()
                    self.focus_x, self.focus_y = demo_default_focus_world(self.engine.paging)
                    self.controller_debug_cycle = 0
                    self.controller_debug_label = None
                    self.controller_debug_saved_state = self.engine.serialize_controller_state()["controller_state"]
                    self.controller_debug_dirty = self.controller_debug_enabled
                elif key in (ord("W"), ord("w")):
                    self._move_focus(0, -self.engine.paging.tile_size)
                elif key in (ord("S"), ord("s")):
                    self._move_focus(0, self.engine.paging.tile_size)
                elif key in (ord("A"), ord("a")):
                    self._move_focus(-self.engine.paging.tile_size, 0)
                elif key in (ord("D"), ord("d")):
                    self._move_focus(self.engine.paging.tile_size, 0)

        def close(self) -> None:  # pragma: no cover
            self.http.stop()
            with self.engine.state_lock:
                self.engine.close()
            close = getattr(super(), "close", None)
            if callable(close):
                close()

        def _refresh_status_title(self) -> None:
            focus_probe_label = format_demo_focus_probe(
                self.engine,
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                debug_view=self.debug_view,
                gas_species=self.gas_view_species,
                light_type=self.optics_view_light,
            )
            title = format_demo_status_title(
                debug_view=self.debug_view,
                brush_mode=self.brush_mode,
                brush_radius=self.brush_radius,
                selected_material=self.selected_material,
                gas_species=self.gas_view_species,
                light_type=self.optics_view_light,
                paused=self.state.paused,
                speed=self.state.speed,
                http_port=self.http.port,
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                world_origin_x=self.engine.paging.origin_x,
                world_origin_y=self.engine.paging.origin_y,
                focus_probe_label=focus_probe_label,
                controller_label=self.controller_debug_label,
            )
            if title == self._status_title:
                return
            now = _time.perf_counter()
            last_update = float(getattr(self, "_last_status_title_update_time", 0.0))
            if self._status_title and now - last_update < 0.5:
                return
            self._status_title = title
            self._last_status_title_update_time = now
            try:
                self.wnd.title = title
            except Exception:
                pass

        def _paint_from_screen(self, x: int, y: int, *, dx: int, dy: int) -> None:
            with self.engine.state_lock:
                bind_gpu_context = getattr(self, "_bind_gui_gpu_context", None)
                if callable(bind_gpu_context):
                    bind_gpu_context()
                world_x, world_y = demo_screen_to_world_cell(
                    x,
                    y,
                    screen_width=self.wnd.width,
                    screen_height=self.wnd.height,
                    active_width=self.demo_visible_width,
                    active_height=self.demo_visible_height,
                    world_origin_x=self.engine.paging.origin_x,
                    world_origin_y=self.engine.paging.origin_y,
                )
                paint_key = (
                    self.brush_mode,
                    int(world_x),
                    int(world_y),
                    int(self.brush_radius),
                    self.selected_material,
                    self.gas_view_species,
                    self.optics_view_light,
                    int(dx),
                    int(dy),
                )
                if paint_key == getattr(self, "_last_paint_key", None):
                    return
                self._last_paint_key = paint_key
                queue_demo_paint(
                    self.engine,
                    self.brush_mode,
                    x=world_x,
                    y=world_y,
                    selected_material=self.selected_material,
                    gas_species=self.gas_view_species,
                    light_type=self.optics_view_light,
                    radius=self.brush_radius,
                    dx=dx,
                    dy=dy,
                )

        def _focus_from_screen(self, x: int, y: int) -> None:
            with self.engine.state_lock:
                self._last_paint_key = None
                bind_gpu_context = getattr(self, "_bind_gui_gpu_context", None)
                if callable(bind_gpu_context):
                    bind_gpu_context()
                world_x, world_y = demo_screen_to_world_cell(
                    x,
                    y,
                    screen_width=self.wnd.width,
                    screen_height=self.wnd.height,
                    active_width=self.demo_visible_width,
                    active_height=self.demo_visible_height,
                    world_origin_x=self.engine.paging.origin_x,
                    world_origin_y=self.engine.paging.origin_y,
                )
                self.focus_x = int(world_x)
                self.focus_y = int(world_y)
                self.engine.advance_paging(self.focus_x, self.focus_y, immediate=True)
                self.controller_debug_dirty = self.controller_debug_enabled

        def _set_controller_debug_enabled(self, enabled: bool) -> None:
            enabled = bool(enabled)
            if enabled == self.controller_debug_enabled:
                return
            self.controller_debug_enabled = enabled
            self.controller_debug_label = None
            self.controller_debug_dirty = enabled
            self.controller_debug_cycle = 0
            if enabled:
                self.controller_debug_saved_state = self.engine.serialize_controller_state()["controller_state"]
                return

            surviving_entities = [
                replace(entity)
                for entity_id, entity in sorted(self.engine.entity_states.items())
                if entity_id != DEMO_CONTROLLER_ENTITY_ID
            ]
            self.engine.sync_entity_states(surviving_entities, immediate=True)
            current_state = self.engine.serialize_controller_state()["controller_state"]
            if isinstance(current_state, dict) and current_state.get("mode") == "demo_probe":
                self.engine.set_controller_state(self.controller_debug_saved_state)
            self.controller_debug_saved_state = None

        def _run_demo_controller_cycle(self, *, apply_turn: bool) -> None:
            entities = build_demo_controller_entities_for_world_focus(
                list(self.engine.entity_states.values()),
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                paging=self.engine.paging,
            )
            controller_state = build_demo_controller_state(
                focus_x=self.focus_x,
                focus_y=self.focus_y,
                cycle=self.controller_debug_cycle,
            )
            cycle = self.engine.run_entity_controller_cycle(
                apply_turn=apply_turn,
                controller_state=controller_state,
                controller_state_provided=True,
                focus_center=(self.focus_x, self.focus_y),
                entities=entities,
            )
            status_label = format_demo_controller_status(
                preview=cycle["preview"],
                turn=cycle["result"],
                applied=cycle["applied"],
            )
            observation_summary = format_demo_controller_observation_summary(
                (cycle["result"] or cycle["preview"]).get("consumed", {}).get("observations"),
                gas_species=self.gas_view_species,
                light_type=self.optics_view_light,
                material_name_by_id=self.engine,
            )
            self.controller_debug_label = (
                f"{status_label} {observation_summary}"
                if status_label is not None and observation_summary is not None
                else status_label or observation_summary
            )
            if cycle["applied"]:
                self.controller_debug_cycle += 1
            self.controller_debug_dirty = False

        def _move_focus(self, dx: int, dy: int) -> None:
            with self.engine.state_lock:
                self.focus_x += dx
                self.focus_y += dy
                self.engine.advance_paging(self.focus_x, self.focus_y, immediate=True)
                self.controller_debug_dirty = self.controller_debug_enabled

    mglw.run_window_config(EngineDemo)


if __name__ == "__main__":  # pragma: no cover
    main()
