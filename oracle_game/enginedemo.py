from __future__ import annotations

import threading
import time as _time
from dataclasses import replace
from typing import Any

import numpy as np

from oracle_game.demo_input import (
    BRUSH_MODE_CYCLE_KEYS,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    BRUSH_MODES,
    CONTROLLER_TOGGLE_KEYS,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    DEBUG_VIEW_KEYMAP,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    DEMO_FOCUS_SCROLL_CELLS_PER_SECOND,
    GAS_SPECIES_CYCLE_KEYS,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    MATERIAL_KEYS,
    OPTICS_LIGHT_CYCLE_KEYS,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    RESET_WORLD_KEYS,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    _alpha_keys,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    advance_demo_focus_scroll,
    apply_demo_paint,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    clamp_demo_brush_radius,
    cycle_demo_brush_mode,
    cycle_demo_named_choice,
    demo_debug_view_for_key,
    demo_focus_scroll_direction,
    demo_focus_scroll_key_direction,
    demo_force_direction_from_drag,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    demo_light_direction_and_spread_from_drag,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    demo_material_for_key,
    demo_screen_to_buffer_cell,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    demo_screen_to_world_cell,
    demo_velocity_from_drag,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    is_demo_brush_cycle_key,
    is_demo_controller_toggle_key,
    is_demo_gas_cycle_key,
    is_demo_optics_cycle_key,
    is_demo_reset_key,
    queue_demo_paint,
    resolve_demo_paint_command,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
)
from oracle_game.demo_render import (
    DEMO_AMBIENT_BOTTOM_LIGHT,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    DEMO_AMBIENT_TOP_LIGHT,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    DEMO_PATTERN_SCALE,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    apply_demo_render_uniforms,
    build_demo_render_uniforms,
)
from oracle_game.demo_sizing import (
    DEMO_ACTIVE_SCALE,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    DEMO_CONTROLLER_ENTITY_ID,
    DEMO_LOGICAL_WORLD_SCALE,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    DEMO_TARGET_CELL_PIXELS,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    ENGINE_DEMO_TITLE,
    _demo_shadow_material_name,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    _format_demo_probe_rgb,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    _format_demo_probe_vector,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    build_demo_controller_entities,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    build_demo_controller_entities_for_world_focus,
    build_demo_controller_probe_entity,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    build_demo_controller_state,
    compute_demo_grid_sizing,
    demo_backend_report,
    demo_brush_selection_label,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    demo_default_focus_world,
    demo_display_material_name,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    demo_view_focus_label,  # noqa: F401  # facade re-export, consumed via oracle_game.enginedemo
    format_demo_controller_observation_summary,
    format_demo_controller_status,
    format_demo_focus_probe,
    format_demo_status_title,
    request_demo_redraw,
)
from oracle_game.http_console import EngineHTTPConsole, EngineRunState
from oracle_game.types import DebugView
from oracle_game.world import WorldEngine

DEMO_REALTIME_BUDGET_CELL_THRESHOLD = 1_000_000
DEMO_FOCUS_KEY_ACTION_RELEASE = 0
DEMO_FOCUS_KEY_ACTION_PRESS = 1
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


def _demo_pace_frame(demo: Any, frame_start: float, frame_time: float) -> tuple[float, float]:
    last_present_time = float(getattr(demo, "_last_present_time", 0.0))
    if last_present_time > 0.0:
        target_frame_time = 1.0 / 60.0
        elapsed_since_present = frame_start - last_present_time
        sleep_time = target_frame_time - elapsed_since_present
        if sleep_time > 0.0:
            _time.sleep(sleep_time)
            frame_start = _time.perf_counter()
        frame_time = max(0.0, frame_start - last_present_time)
        demo._last_present_time = frame_start
    return frame_start, frame_time


def _demo_step_simulation(demo: Any, frame_time: float) -> bool:
    stepped = False
    steps = 0
    sim_start = _time.perf_counter()
    if not demo.state.paused:
        demo.accumulator += frame_time * max(0.1, demo.state.speed)
        demo.accumulator = min(demo.accumulator, 2.0 / 60.0)
        while demo.accumulator >= 1.0 / 60.0 and steps < 1:
            demo.engine.step(1.0 / 60.0)
            demo.accumulator -= 1.0 / 60.0
            steps += 1
            stepped = True
    elif demo.state.single_step:
        demo.engine.step(1.0 / 60.0)
        demo.state.single_step = False
        steps = 1
        stepped = True
    sim_done = _time.perf_counter()
    demo.sim_ms = (sim_done - sim_start) * 1000.0
    demo._record_gpu_steps(steps, sim_done)
    return stepped


def _demo_sync_display_textures(demo: Any, now: float) -> float:
    gpu_debug_synced = False
    gas_species_id = -1
    light_dose_channel = -1
    if demo.debug_view != DebugView.MATERIAL:
        if demo.debug_view == DebugView.GAS:
            gas_species_id = demo.engine._resolve_sanctioned_gas_id(demo.gas_view_species)
        if (
            demo.debug_view in {DebugView.LIGHT, DebugView.OPTICS}
            and demo.optics_view_light is not None
        ):
            light_id = demo.engine._resolve_sanctioned_light_id(demo.optics_view_light)
            dose_channel = (
                demo.engine._shadow_light_dose_channel(light_id) if light_id >= 0 else None
            )
            light_dose_channel = -1 if dose_channel is None else int(dose_channel)
    sync_start = _time.perf_counter()
    if demo.debug_view == DebugView.MATERIAL:
        sync_display = getattr(demo.engine.bridge, "sync_display_textures", None)
        if callable(sync_display):
            sync_display(demo.engine)
    else:
        gpu_backend = getattr(demo.engine, "simulation_backend", "") == "gpu"
        sync_debug_display = getattr(demo.engine.bridge, "sync_debug_display_texture", None)
        if callable(sync_debug_display):
            gpu_debug_synced = bool(
                sync_debug_display(
                    demo.engine,
                    view=demo.debug_view.value,
                    gas_species_id=gas_species_id,
                    light_dose_channel=light_dose_channel,
                )
            )
        allow_cpu_debug_upload = not gpu_backend or not callable(sync_debug_display)
        if not gpu_debug_synced and allow_cpu_debug_upload:
            _demo_upload_cpu_debug_frame(demo, now)
    sync_done = _time.perf_counter()
    demo.sync_ms = (sync_done - sync_start) * 1000.0
    return sync_done


def _demo_upload_cpu_debug_frame(demo: Any, now: float) -> None:
    debug_key = (demo.debug_view, demo.gas_view_species, demo.optics_view_light)
    debug_refresh_due = (
        getattr(demo, "_debug_frame_cache", None) is None
        or getattr(demo, "_debug_frame_cache_key", None) != debug_key
        or now - float(getattr(demo, "_last_debug_texture_upload_time", 0.0))
        >= DEMO_DEBUG_TEXTURE_REFRESH_SECONDS
    )
    if not debug_refresh_due:
        return
    debug = demo.engine.debug_frame(
        demo.debug_view,
        gas_species=demo.gas_view_species,
        light_type=demo.optics_view_light,
    )
    demo._debug_frame_cache = debug
    demo._debug_frame_cache_key = debug_key
    demo._last_debug_texture_upload_time = now
    try:
        demo.engine.bridge.sync_world(
            demo.engine,
            debug_frame=debug,
            upload_debug_texture=True,
        )
    except TypeError as exc:
        if "upload_debug_texture" not in str(exc):
            raise
        demo.engine.bridge.sync_world(demo.engine, debug_frame=debug)


def _demo_draw_frame(demo: Any) -> float:
    apply_demo_render_uniforms(
        demo.program,
        build_demo_render_uniforms(demo.engine, debug_view=demo.debug_view),
    )
    demo.engine.bridge.texture("material").use(0)
    demo.engine.bridge.texture("light").use(1)
    demo.engine.bridge.texture("debug").use(2)
    demo.engine.bridge.atlas_texture().use(3)
    demo.ctx.clear(0.02, 0.03, 0.05)
    demo.vao.render(mode=demo.ctx.TRIANGLE_STRIP)
    return _time.perf_counter()


def _demo_publish_runtime_state(demo: Any) -> None:
    demo.engine.demo_runtime_state = {
        "frame_id": int(demo.engine.frame_id),
        "debug_view": demo.debug_view.value,
        "force_debug_texture": demo.debug_view != DebugView.MATERIAL,
        "visible_size": [int(demo.demo_visible_width), int(demo.demo_visible_height)],
        "active_size": [
            int(demo.engine.paging.active_width),
            int(demo.engine.paging.active_height),
        ],
        "buffer_size": [int(demo.engine.width), int(demo.engine.height)],
        "logical_world_size": [
            int(demo.demo_logical_world_width),
            int(demo.demo_logical_world_height),
        ],
        "origin": [int(demo.engine.paging.origin_x), int(demo.engine.paging.origin_y)],
        "buffer_origin": [
            int(demo.engine.paging.buffer_origin_x),
            int(demo.engine.paging.buffer_origin_y),
        ],
        "cpu_fps": float(getattr(demo, "cpu_fps", 0.0)),
        "gpu_fps": float(getattr(demo, "gpu_fps", 0.0)),
        "frame_ms": float(demo.frame_ms),
        "sim_ms": float(demo.sim_ms),
        "sync_ms": float(demo.sync_ms),
        "render_ms": float(demo.render_ms),
        "scroll_ms": float(getattr(demo, "scroll_ms", 0.0)),
        "controller_ms": float(getattr(demo, "controller_ms", 0.0)),
        "worst_frames": list(getattr(demo, "_frame_stage_history", [])),
        "backend_report": demo_backend_report(demo.engine),
    }
    demo._refresh_status_title()
    request_demo_redraw(demo.wnd)


def _demo_record_frame_stage(demo: Any, **stage: float) -> None:
    history = getattr(demo, "_frame_stage_history", None)
    if history is None:
        history = []
        demo._frame_stage_history = history
    history.append(stage)
    # Keep the worst frames by total frame_ms for the HTTP console dump.
    history.sort(key=lambda s: s["frame_ms"], reverse=True)
    del history[8:]


def _demo_handle_brush_and_view_keys(demo: Any, key: int) -> bool:
    if (material := demo_material_for_key(key)) is not None:
        demo.selected_material = material
    elif (debug_view := demo_debug_view_for_key(key)) is not None:
        demo.debug_view = debug_view
        demo._debug_frame_cache = None
        demo._debug_frame_cache_key = None
    elif key == ord("["):
        demo.brush_radius = clamp_demo_brush_radius(demo.brush_radius - 1)
    elif key == ord("]"):
        demo.brush_radius = clamp_demo_brush_radius(demo.brush_radius + 1)
    else:
        return False
    return True


def _demo_handle_speed_and_step_keys(demo: Any, key: int) -> bool:
    if key == ord("-"):
        demo.state.speed = max(0.1, demo.state.speed * 0.8)
    elif key == ord("="):
        demo.state.speed = min(8.0, demo.state.speed * 1.25)
    elif key == ord(" "):
        demo.state.paused = not demo.state.paused
    elif key in (ord("N"), ord("n")):
        demo.state.single_step = True
    else:
        return False
    return True


def _demo_handle_cycle_and_reset_keys(demo: Any, key: int) -> bool:
    if is_demo_optics_cycle_key(key):
        demo.optics_view_light = cycle_demo_named_choice(
            demo.optics_view_light,
            (None, *demo.engine.rulebook.lights_by_name.keys()),
        )
    elif is_demo_gas_cycle_key(key):
        gas_choices = tuple(name for name in demo.engine.gas_name_by_id if name)
        next_species = cycle_demo_named_choice(demo.gas_view_species, gas_choices)
        if next_species is not None:
            demo.gas_view_species = next_species
    elif is_demo_brush_cycle_key(key):
        demo.brush_mode = cycle_demo_brush_mode(demo.brush_mode)
    elif is_demo_controller_toggle_key(key):
        demo._set_controller_debug_enabled(not demo.controller_debug_enabled)
    elif is_demo_reset_key(key):
        demo.engine.reset_world()
        demo.focus_x, demo.focus_y = demo_default_focus_world(demo.engine.paging)
        demo._focus_scroll_x = 0.0
        demo._focus_scroll_y = 0.0
        demo.controller_debug_cycle = 0
        demo.controller_debug_label = None
        demo.controller_debug_saved_state = demo.engine.serialize_controller_state()[
            "controller_state"
        ]
        demo.controller_debug_dirty = demo.controller_debug_enabled
    else:
        return False
    return True


def _demo_key_action_is_release(action: object) -> bool:
    """moderngl-window passes release as int 0 on some backends, 'ACTION_RELEASE' on others."""
    if isinstance(action, str):
        return action == "ACTION_RELEASE"
    return int(action) == DEMO_FOCUS_KEY_ACTION_RELEASE


def _demo_handle_focus_move_keys(demo: Any, key: int, action: int) -> bool:
    """Track WASD hold state; the render loop scrolls focus continuously."""
    if demo_focus_scroll_key_direction(key) is None:
        return False
    held = getattr(demo, "_focus_scroll_held_keys", None)
    if held is None:
        held = set()
        demo._focus_scroll_held_keys = held
    if _demo_key_action_is_release(action):
        held.discard(int(key))
        if not held:
            # Drop the sub-tile remainder so a fresh hold always starts aligned.
            demo._focus_scroll_x = 0.0
            demo._focus_scroll_y = 0.0
        return True
    held.add(int(key))
    return True


def _demo_scroll_held_focus_keys(demo: Any, frame_time: float) -> bool:
    """Advance focus by the held WASD keys; paging triggers via its own deadzone."""
    direction = demo_focus_scroll_direction(getattr(demo, "_focus_scroll_held_keys", set()))
    if direction is None:
        return False
    scroll_x, scroll_y, step_x, step_y = advance_demo_focus_scroll(
        float(getattr(demo, "_focus_scroll_x", 0.0)),
        float(getattr(demo, "_focus_scroll_y", 0.0)),
        direction=direction,
        frame_time=frame_time,
        speed_cells_per_second=float(
            getattr(demo, "focus_scroll_speed", DEMO_FOCUS_SCROLL_CELLS_PER_SECOND)
        ),
        tile_size=int(demo.engine.paging.tile_size),
    )
    demo._focus_scroll_x = scroll_x
    demo._focus_scroll_y = scroll_y
    if step_x == 0 and step_y == 0:
        return False
    demo.focus_x += step_x
    demo.focus_y += step_y
    demo.engine.advance_paging(demo.focus_x, demo.focus_y, immediate=True)
    demo.controller_debug_dirty = demo.controller_debug_enabled
    return True


def main() -> None:
    try:
        import moderngl  # noqa: F401
        import moderngl_window as mglw
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install moderngl and moderngl-window inside the venv before running enginedemo."
        ) from exc

    # Prefer the discrete NVIDIA GPU on PRIME hybrid-graphics laptops/desktops.
    # These must be set before pyglet opens the X11/GLX window, which happens
    # inside run_window_config below.
    import os

    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    class EngineDemo(mglw.WindowConfig):
        gl_version = (4, 3)
        title = ENGINE_DEMO_TITLE
        window_size = (1440, 900)
        vsync = False
        resizable = True
        aspect_ratio = None

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._init_world_engine()
            self._init_http_console()
            self._init_demo_ui_state()
            self._init_perf_counters()
            self._build_render_resources()
            self._prime_render_textures()
            self._refresh_status_title()

        def _init_world_engine(self) -> None:
            ctx_info = getattr(self.ctx, "info", None)
            renderer = str(ctx_info.get("GL_RENDERER", "")) if hasattr(ctx_info, "get") else ""
            if (
                os.environ.get("__NV_PRIME_RENDER_OFFLOAD") == "1"
                and "NVIDIA" not in renderer.upper()
            ):
                import warnings

                warnings.warn(
                    f"enginedemo requested the discrete NVIDIA GPU but the GL context landed on "
                    f"{renderer!r}; simulation will run on the wrong (slow) device. "
                    f"Check your PRIME/offload setup.",
                    stacklevel=2,
                )
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

        def _init_http_console(self) -> None:
            self.state = EngineRunState()
            try:
                self.http = EngineHTTPConsole(self.engine, self.state, own_gpu_context=False)
            except TypeError as exc:
                if "own_gpu_context" not in str(exc):
                    raise
                self.http = EngineHTTPConsole(self.engine, self.state)
                setattr(self.http, "own_gpu_context", False)
            self.http.start()

        def _init_demo_ui_state(self) -> None:
            self.brush_radius = 3
            self.selected_material = MATERIAL_KEYS[0]
            self.brush_mode = BRUSH_MODES[0]
            self.debug_view = DebugView.MATERIAL
            self.gas_view_species = "water_gas"
            self.optics_view_light: str | None = None
            self.focus_x, self.focus_y = demo_default_focus_world(self.engine.paging)
            self.focus_scroll_speed = DEMO_FOCUS_SCROLL_CELLS_PER_SECOND
            self._focus_scroll_held_keys: set[int] = set()
            self._focus_scroll_x = 0.0
            self._focus_scroll_y = 0.0
            self.accumulator = 0.0
            self._last_present_time = _time.perf_counter()
            self._status_title = ""
            self.controller_debug_enabled = False
            self.controller_debug_cycle = 0
            self.controller_debug_dirty = False
            self.controller_debug_label: str | None = None
            self.controller_debug_saved_state: Any = None

        def _init_perf_counters(self) -> None:
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
            frame_start, frame_time = _demo_pace_frame(self, frame_start, frame_time)
            bind_gpu_context = getattr(self, "_bind_gui_gpu_context", None)
            if callable(bind_gpu_context):
                bind_gpu_context()
            now = _time.perf_counter()
            self._record_cpu_frame(now)
            with self.engine.state_lock:
                scrolled = _demo_scroll_held_focus_keys(self, frame_time)
                scroll_done = _time.perf_counter()
                stepped = _demo_step_simulation(self, frame_time)
                step_done = _time.perf_counter()
                if self.controller_debug_enabled and (self.controller_debug_dirty or stepped):
                    self._run_demo_controller_cycle(apply_turn=stepped)
                controller_done = _time.perf_counter()
                self.engine.default_debug_view = self.debug_view
                sync_done = _demo_sync_display_textures(self, now)
                render_done = _demo_draw_frame(self)
                # Per-stage breakdown for diagnosing frame spikes (all ms).
                self.scroll_ms = (scroll_done - now) * 1000.0
                self.controller_ms = (controller_done - step_done) * 1000.0
                self.render_ms = (render_done - sync_done) * 1000.0
                self.frame_ms = (render_done - frame_start) * 1000.0
                _demo_record_frame_stage(
                    self,
                    scrolled=scrolled,
                    stepped=stepped,
                    scroll_ms=self.scroll_ms,
                    sim_ms=self.sim_ms,
                    controller_ms=self.controller_ms,
                    sync_ms=self.sync_ms,
                    render_ms=self.render_ms,
                    frame_ms=self.frame_ms,
                )
                _demo_publish_runtime_state(self)

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
            with self.engine.state_lock:
                if _demo_handle_focus_move_keys(self, key, action):
                    return
                if _demo_key_action_is_release(action):
                    return
                if _demo_handle_brush_and_view_keys(self, key):
                    return
                if _demo_handle_speed_and_step_keys(self, key):
                    return
                if _demo_handle_cycle_and_reset_keys(self, key):
                    return

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
                cpu_fps=self.cpu_fps,
                gpu_fps=self.gpu_fps,
                frame_ms=self.frame_ms,
                sim_ms=self.sim_ms,
                sync_ms=self.sync_ms,
                render_ms=self.render_ms,
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
                self._focus_scroll_x = 0.0
                self._focus_scroll_y = 0.0
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
                self.controller_debug_saved_state = self.engine.serialize_controller_state()[
                    "controller_state"
                ]
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

    mglw.run_window_config(EngineDemo)


if __name__ == "__main__":  # pragma: no cover
    main()
