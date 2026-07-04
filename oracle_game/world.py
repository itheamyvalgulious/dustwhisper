from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from enum import Enum
import json
import threading
import time
from typing import Any, Iterable, Sequence
import zlib

import numpy as np

from oracle_game.active import ActiveRegionTracker
from oracle_game.gpu import (
    FALLING_ISLAND_BREAK_KIND_IDS,
    GPUBridge,
    GPUBufferReadbackSource,
    GPUCellCoreWindowReadbackSource,
    GPUGasWindowReadbackSource,
    GPUReadbackSegment,
    GPUSegmentedBufferReadbackSource,
    GPUSegmentedCellCoreWindowReadbackSource,
    GPUSegmentedTextureReadbackSource,
    GPUTextureReadbackSource,
    LIQUID_SOLVER_KIND_IDS,
    PAGE_STRIPE_AXIS_IDS,
    PAGE_STRIPE_FIELD_PATHS,
    PAGE_STRIPE_KIND_IDS,
    POWDER_SOLVER_KIND_IDS,
    typed_gas_id,
    typed_material_id,
    typed_light_id,
    unpack_cell_core,
)
from oracle_game.readback_contract import READBACK_ALLOWED_CHANNELS, READBACK_CHANNEL_BITS
from oracle_game.page_store import InMemoryPageStore, PageStore, StoredStripeKey
from oracle_game.paging import RingPagingWindow
from oracle_game.rules import RuleBook, build_default_optics_entries, build_default_payloads
from oracle_game.sim.collapse import CollapseSolver
from oracle_game.sim.gas import GasSolver
from oracle_game.sim.gpu_collapse_dirty import (
    clear_collapse_structure_dirty_tile_mask,
    drain_collapse_structure_dirty_tile_regions,
)
from oracle_game.sim.heat import HeatSolver
from oracle_game.sim.liquid import LiquidSolver
from oracle_game.sim.motion import MotionSolver
from oracle_game.sim.optics import OpticsSolver
from oracle_game.sim.gpu_placeholders import GPUPlaceholderPipeline
from oracle_game.sim.gpu_page_stripes import GPUPageStripePipeline
from oracle_game.sim.gpu_world_commands import COMMAND_KIND_IDS as GPU_WORLD_COMMAND_KIND_IDS
from oracle_game.sim.gpu_world_commands import GPUWorldCommandPipeline
from oracle_game.sim.reactions import ReactionSolver
from oracle_game.types import (
    CellFlag,
    CarrierIntent,
    ChangeIntent,
    COLLAPSE_BEHAVIOR_IDS,
    DebugView,
    Direction,
    EntityCellFeedback,
    EntityFeedback,
    EntityObservationSpec,
    EntityPlaceholder,
    EntityStatePatch,
    EntityState,
    FallingIslandRecord,
    ForceSource,
    GasSpeciesDef,
    LightTypeDef,
    MaterialDef,
    MaterialOpticsDef,
    ObservationResult,
    ObservationTarget,
    PageStripeUpdate,
    PairReactionRule,
    Phase,
    ReactionAction,
    ReactionType,
    ReadbackRequest,
    ReadbackResult,
    ResolvedCarrierIntent,
    ResolvedChangeIntent,
    ResolvedTarget,
    SelfReactionRule,
    TargetQuery,
    WorldFrameInput,
    WorldFrameOutput,
    WorldFramePreview,
    WorldCommand,
)

CARDINAL_DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

TERRAIN_ANCHOR_FILTERS = {"empty", "hill", "hole", "liquid", "pool", "solid", "tree", "wall"}
IGNORED_ANCHOR_FILTERS = {"nearest"}
TARGET_QUERY_CELLS_PER_METER = 4.0
TARGET_QUERY_DISTANCE_HINT_CELLS: dict[str, int] = {
    "near": 4,
    "far": 12,
}
UNSET_CONTROLLER_STATE = object()
TARGETED_COMMAND_COORD_FIELDS: dict[str, tuple[str, str]] = {
    "advance_paging": ("center_x", "center_y"),
    "inject_force": ("x", "y"),
    "inject_gas": ("x", "y"),
    "inject_light": ("x", "y"),
    "inject_material": ("x", "y"),
    "inject_temperature": ("x", "y"),
    "inject_velocity": ("x", "y"),
    "request_readback": ("center_x", "center_y"),
    "write_material_region": ("x", "y"),
}
PUBLIC_WORLD_COMMAND_KINDS = (
    "advance_paging",
    "inject_force",
    "inject_gas",
    "inject_light",
    "inject_material",
    "inject_temperature",
    "inject_velocity",
    "patch_entity_states",
    "request_readback",
    "sync_entity_observation_specs",
    "sync_entity_states",
    "write_material_region",
)
PAIR_REACTION_RULE_SET_NAMES = frozenset(
    {
        "material_material",
        "material_gas",
        "material_light",
        "gas_gas",
        "gas_light",
    }
)
REACTION_RULE_SET_NAMES = frozenset({*PAIR_REACTION_RULE_SET_NAMES, "self_rules"})
READBACK_ALLOWED_CHANNEL_SET = frozenset(READBACK_ALLOWED_CHANNELS)
MAX_ASYNC_READBACK_WIDTH = 64
MAX_ASYNC_READBACK_HEIGHT = 64
GPU_REALTIME_BUDGET_CELL_THRESHOLD = 1080 * 720
BASE_MATERIAL_RUNTIME_ALIASES: dict[str, str] = {
    "sand": "sand_powder",
    "gravel": "gravel_powder",
    "soil": "soil_powder",
    "sandstone": "sandstone_solid",
    "raw_stone": "raw_stone_solid",
    "obsidian": "obsidian_solid",
    "root": "root_solid",
    "log": "log_solid",
    "leaf": "leaf_powder",
    "vine": "vine_solid",
    "moss": "moss_powder",
    "iron": "iron_solid",
    "gold": "gold_solid",
    "water": "water_liquid",
    "poison": "poison_liquid",
    "acid": "acid_liquid",
    "oil": "oil_liquid",
    "explosive": "explosive_solid",
    "phosphor_visible": "phosphor_visible_powder",
    "phosphor_holy": "phosphor_holy_powder",
    "phosphor_chaos": "phosphor_chaos_powder",
    "phosphor_magic": "phosphor_magic_powder",
    "pollution": "pollution_powder",
    "vortex_heart": "vortex_heart_solid",
    "fire": "fire_powder",
    "placeholder": "placeholder_solid",
}
ENTITY_STATE_PATCHABLE_FIELDS = frozenset(
    {
        "x",
        "y",
        "width",
        "height",
        "velocity_xy",
        "facing_xy",
        "placeholder_material",
        "tags",
        "observe_channels",
        "observe_pad_cells",
        "observe_width",
        "observe_height",
        "observe_label",
    }
)
ENTITY_STATE_PATCH_METADATA_FIELDS = frozenset({"_world_x", "_world_y"})


class WorldEngine:
    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 192,
        active_width: int | None = None,
        active_height: int | None = None,
        gas_cell_size: int = 4,
        gpu_context: Any | None = None,
        page_store: PageStore | None = None,
        simulation_backend: str = "gpu",
    ) -> None:
        simulation_backend = str(simulation_backend).lower()
        if simulation_backend not in {"gpu", "cpu"}:
            raise ValueError("simulation_backend must be one of: gpu, cpu")
        self.simulation_backend = simulation_backend
        self._world_simulation_frame_active = False
        self.width = width
        self.height = height
        self.gas_cell_size = gas_cell_size
        self.gas_width = max(1, (width + gas_cell_size - 1) // gas_cell_size)
        self.gas_height = max(1, (height + gas_cell_size - 1) // gas_cell_size)
        self.paging = RingPagingWindow(width, height, active_width or width // 2, active_height or height // 2)
        self.active = ActiveRegionTracker(width, height)
        self.rulebook = RuleBook()
        self.bridge = (
            GPUBridge(create_standalone=False)
            if simulation_backend == "cpu" and gpu_context is None
            else GPUBridge(ctx=gpu_context)
        )
        if simulation_backend == "gpu" and not self._gpu_context_available():
            raise RuntimeError("GPU world simulation requires a ModernGL 4.3+ context; CPU fallback is disabled")
        self.page_store = InMemoryPageStore() if page_store is None else page_store
        self.frame_id = 0
        self.state_lock = threading.RLock()
        self.command_queue: deque[WorldCommand] = deque()
        self.pending_frame_inputs: deque[WorldFrameInput] = deque()
        self.completed_frame_outputs: deque[WorldFrameOutput] = deque()
        self.canceled_frame_submission_ids: set[int] = set()
        self.next_frame_submission_id = 1
        self.next_readback_request_id = 1
        self.pending_readbacks: list[ReadbackRequest] = []
        self.inflight_readbacks: list[ReadbackRequest] = []
        self.completed_readbacks: deque[ReadbackResult] = deque()
        self.canceled_readback_request_ids: set[int] = set()
        self.last_entity_observation_consume_snapshot: dict[str, Any] = {
            "frame_id": 0,
            "consumed": 0,
            "consumed_readbacks": [],
            "observations": {},
            "entity_feedback": {},
        }
        self.controller_state_snapshot: Any = None
        self.bootstrap_log: list[str] = []
        self.bridge_frame_commands: list[WorldCommand] = []
        self.bridge_frame_readback_requests: list[ReadbackRequest] = []
        self.bridge_frame_placeholders: list[EntityPlaceholder] = []
        self.bridge_frame_placeholder_dirty_rects: list[tuple[int, int, int, int]] = []
        self._pending_placeholder_dirty_rects: list[tuple[int, int, int, int]] = []
        self.bridge_frame_paging_updates: list[PageStripeUpdate] = []
        self.bridge_frame_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
        self._bridge_inputs_prepared = False
        self._gpu_cpu_dirty_resources: set[str] = set()
        self._resolver_blocked_cells: set[tuple[int, int]] | None = None
        self._resolver_released_cells: set[tuple[int, int]] | None = None
        self.force_sources: list[ForceSource] = []
        self.persistent_emitters: list[dict[str, object]] = []
        self.emitters: list[dict[str, object]] = []
        self._formal_gpu_frame_has_light_dose: bool | None = None
        self._gpu_optics_outputs_clear = False
        self.gpu_realtime_budget_enabled = True
        self.gpu_realtime_budget_cell_threshold = GPU_REALTIME_BUDGET_CELL_THRESHOLD
        self.profile_passes_enabled = False
        self.profile_passes_sync = False
        self.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}, "skipped_stages": []}
        self.last_skipped_gpu_stages: list[str] = []
        self.formal_collapse_interval_frames = 4
        self.collapse_dirty_regions: list[tuple[int, int, int, int]] = []
        self.collapse_deferred_regions: list[tuple[int, int, int, int]] = []
        self._gpu_collapse_structure_dirty_tiles_pending = False
        self._gpu_collapse_structure_dirty_tiles_deferred = False
        self.islands: dict[int, object] = {}
        self.entity_states: dict[int, EntityState] = {}
        self.entity_placeholders: dict[int, set[tuple[int, int]]] = {}
        self.next_island_id = 1

        self.material_id = np.zeros((height, width), dtype=np.int32)
        self.phase = np.zeros((height, width), dtype=np.uint8)
        self.cell_flags = np.zeros((height, width), dtype=np.uint8)
        self.velocity = np.zeros((height, width, 2), dtype=np.float32)
        self.cell_temperature = np.full((height, width), 20.0, dtype=np.float32)
        self.timer_pack = np.zeros((height, width, 4), dtype=np.uint8)
        self.integrity = np.zeros((height, width), dtype=np.float32)
        self.island_id = np.zeros((height, width), dtype=np.int32)
        self.entity_id = np.zeros((height, width), dtype=np.int32)
        self.placeholder_displaced_material = np.zeros((height, width), dtype=np.int32)
        self.collapse_delay_pending = np.zeros((height, width), dtype=np.bool_)

        self.flow_velocity = np.zeros((self.gas_height, self.gas_width, 2), dtype=np.float32)
        self.ambient_temperature = np.full((self.gas_height, self.gas_width), 20.0, dtype=np.float32)
        self.pressure_ping = np.zeros((self.gas_height, self.gas_width), dtype=np.float32)
        self.gas_concentration = np.zeros((1, self.gas_height, self.gas_width), dtype=np.float32)
        self.visible_illumination = np.zeros((height, width, 3), dtype=np.float32)
        self.cell_optical_dose = np.zeros((1, height, width), dtype=np.float32)
        self.gas_optical_dose = np.zeros((1, self.gas_height, self.gas_width), dtype=np.float32)
        self.default_debug_view = DebugView.MATERIAL

        self.material_density = np.zeros(1, dtype=np.float32)
        self.material_base_color = np.zeros((1, 3), dtype=np.float32)
        self.material_gravity = np.zeros(1, dtype=np.float32)
        self.material_wind = np.zeros(1, dtype=np.float32)
        self.material_drag = np.zeros(1, dtype=np.float32)
        self.material_friction = np.zeros(1, dtype=np.float32)
        self.material_elasticity = np.zeros(1, dtype=np.float32)
        self.material_max_dda_step = np.zeros(1, dtype=np.int32)
        self.material_default_phase = np.zeros(1, dtype=np.uint8)
        self.material_base_integrity = np.zeros(1, dtype=np.float32)
        self.material_spawn_temperature = np.full(1, np.nan, dtype=np.float32)
        self.material_reaction_slots = np.full((1, 8), -1, dtype=np.int32)
        self.material_material_tag_mask = np.zeros(1, dtype=np.uint32)
        self.material_gas_tag_mask = np.zeros(1, dtype=np.uint32)
        self.material_light_tag_mask = np.zeros(1, dtype=np.uint32)
        self.material_powder_solver_kind = np.zeros(1, dtype=np.uint8)
        self.material_liquid_solver_kind = np.zeros(1, dtype=np.uint8)
        self.material_falling_island_break_kind = np.zeros(1, dtype=np.uint8)
        self.material_heat_capacity = np.zeros(1, dtype=np.float32)
        self.material_conductivity = np.zeros(1, dtype=np.float32)
        self.material_ambient_exchange = np.zeros(1, dtype=np.float32)
        self.material_is_structural = np.zeros(1, dtype=np.bool_)
        self.material_is_support_anchor = np.zeros(1, dtype=np.bool_)
        self.material_is_plant = np.zeros(1, dtype=np.bool_)
        self.material_is_placeholder = np.zeros(1, dtype=np.bool_)
        self.material_collapse_behavior = np.zeros(1, dtype=np.uint8)
        self.material_collapse_generation_id = np.zeros(1, dtype=np.int32)
        self.material_powder_generation_id = np.zeros(1, dtype=np.int32)
        self.material_name_by_id: list[str] = [""]
        self.tag_bits_by_name: dict[str, int] = {}
        self.random_convert_material_ids: list[int] = []
        self.placeholder_material_id = 0

        self.gas_material_reaction_tag_mask = np.zeros(1, dtype=np.uint32)
        self.gas_light_reaction_tag_mask = np.zeros(1, dtype=np.uint32)
        self.gas_density_factor = np.zeros(1, dtype=np.float32)
        self.gas_condense_material_id = np.zeros(1, dtype=np.int32)
        self.gas_name_by_id: list[str] = []
        self.air_gas_species_id = -1

        self.light_default_range = np.zeros(1, dtype=np.int32)
        self.light_dose_channel = np.zeros(1, dtype=np.int32)
        self.light_color = np.zeros((1, 3), dtype=np.float32)
        self.light_name_by_id: list[str] = [""]
        self._stable_shadow_payloads: dict[str, Any] = {}

        self.gas_solver = GasSolver()
        self.heat_solver = HeatSolver()
        self.collapse_solver = CollapseSolver()
        self.motion_solver = MotionSolver()
        self.liquid_solver = LiquidSolver()
        self.optics_solver = OpticsSolver()
        self.reaction_solver = ReactionSolver()
        self.placeholder_pipeline = GPUPlaceholderPipeline()
        self.page_stripe_pipeline = GPUPageStripePipeline()
        self.grid_command_pipeline = GPUWorldCommandPipeline()

        self.bootstrap_defaults()
        self.reset_world()
        self.bridge.sync_world(self)
        if self.simulation_backend == "gpu":
            self.bridge.mark_gpu_authoritative(
                "cell_core",
                "material",
                "island_id",
                "entity_id",
                "placeholder_displaced_material",
                "collapse_delay_pending",
                "gas_concentration",
                "ambient_temperature",
                "flow_velocity",
                "pressure_ping",
                "visible_illumination",
                "cell_optical_dose",
                "gas_optical_dose",
                "active_meta",
                "active_tile_ttl",
                "active_chunk_mask",
            )
            self._gpu_cpu_dirty_resources.clear()
        self._closed = False

    def use_cpu_oracle_backend(self) -> None:
        self.simulation_backend = "cpu"

    def require_gpu_world_backend(self) -> None:
        self.simulation_backend = "gpu"

    def prewarm_formal_connected_collapse(self) -> bool:
        if self.simulation_backend != "gpu":
            return False
        pipeline = self.collapse_solver.gpu_pipeline
        if not pipeline.available(self):
            return False
        pipeline.prewarm_formal_connected_resources(self)
        return True

    def _gpu_context_available(self) -> bool:
        ctx = self.bridge.ctx
        return bool(self.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)

    def _gpu_world_simulation_required(self) -> bool:
        return self.simulation_backend == "gpu"

    def _gpu_realtime_budget_active(self) -> bool:
        if not (self.gpu_realtime_budget_enabled and self.simulation_backend == "gpu"):
            return False
        active_tile_count = self._gpu_active_tile_count()
        if active_tile_count <= 0:
            return False
        estimated_active_cells = active_tile_count * int(self.active.tile_size) * int(self.active.tile_size)
        return estimated_active_cells >= int(self.gpu_realtime_budget_cell_threshold)

    def _gpu_active_tile_count(self) -> int:
        if "active_tile_ttl" in self.bridge.gpu_authoritative_resources:
            active_meta = self.bridge.shadow_buffers.get("active_meta")
            if isinstance(active_meta, np.ndarray) and active_meta.size > 0:
                return int(active_meta[0]["active_tile_count"])
            return 0
        active_tile_ttl = np.asarray(self.active.active_tile_ttl, dtype=np.int32)
        if active_tile_ttl.size <= 0:
            return 0
        return int(np.count_nonzero(active_tile_ttl > 0))

    def _skip_budgeted_gpu_stage(self, stage: str) -> bool:
        return False

    def _should_run_formal_collapse_this_frame(self) -> bool:
        if self.simulation_backend != "gpu":
            return True
        interval = max(1, int(getattr(self, "formal_collapse_interval_frames", 1)))
        if interval <= 1:
            return True
        frame_id = max(1, int(getattr(self, "frame_id", 1)))
        return (frame_id - 1) % interval == 0

    @contextmanager
    def _profile_pass(self, name: str):
        profile = self.last_pass_profile if self.profile_passes_enabled else None
        ctx = self.bridge.ctx if bool(getattr(self, "profile_passes_sync", False)) else None
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

    def _gpu_pipeline_available(self, pipeline: Any, name: str, *, require: bool | None = None) -> bool:
        if self.simulation_backend == "cpu":
            return False
        available = bool(pipeline.available(self))
        required = self._gpu_world_simulation_required() if require is None else bool(require)
        if required and not available:
            raise RuntimeError(f"GPU world simulation requires the {name} GPU pipeline; CPU fallback is disabled")
        return available

    def _require_gpu_stage(self, name: str) -> None:
        if self._gpu_world_simulation_required():
            raise RuntimeError(f"GPU world simulation requires GPU support for {name}; CPU fallback is disabled")

    def _require_gpu_authoritative_resources(self, stage: str, *resource_names: str) -> None:
        if not (self.simulation_backend == "gpu" and self._world_simulation_frame_active):
            return
        missing = [str(name) for name in resource_names if str(name) not in self.bridge.gpu_authoritative_resources]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"GPU world simulation requires GPU-authoritative {stage} resources: {joined}; "
                "CPU fallback is disabled"
            )

    def _require_cpu_oracle_backend(self, name: str) -> None:
        if self.simulation_backend != "cpu":
            raise RuntimeError(
                f"{name} CPU oracle path requires simulation_backend='cpu'; CPU fallback is disabled"
            )

    def _invalidate_gpu_authoritative_resources(self, *resource_names: str) -> None:
        if self.simulation_backend == "gpu":
            self.bridge.clear_gpu_authoritative(*resource_names)
            self._bridge_inputs_prepared = False
            if not self._world_simulation_frame_active:
                self._gpu_cpu_dirty_resources.update(str(name) for name in resource_names)

    def _invalidate_gpu_authoritative_cell_resources(self) -> None:
        self._invalidate_gpu_authoritative_resources(
            "cell_core",
            "material",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_delay_pending",
            "liquid_flow_intent",
        )

    def bootstrap_defaults(self) -> None:
        payloads = build_default_payloads()
        self.update_material_table(payloads["materials"])
        self.update_gas_species_table(payloads["gases"])
        self.update_light_type_table(payloads["lights"])
        self.update_material_optics_table(payloads["optics"])
        self.update_reaction_table(payloads["actions"], payloads["rules"])

    def _material_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return [asdict(material) for _, material in sorted(self.rulebook.materials_by_id.items())]

    def _gas_species_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return [asdict(gas) for _, gas in sorted(self.rulebook.gases_by_id.items())]

    def _light_type_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return [asdict(light) for _, light in sorted(self.rulebook.lights_by_id.items())]

    def _material_optics_table_snapshot_payload(self) -> list[dict[str, Any]]:
        return [asdict(entry) for _, entry in sorted(self.rulebook.optics.items())]

    def _material_optics_snapshot_map(self) -> dict[tuple[str, str], MaterialOpticsDef]:
        payload = self._stable_shadow_payload("optics", self._material_optics_table_snapshot_payload)
        snapshot: dict[tuple[str, str], MaterialOpticsDef] = {}
        for item in payload:
            entry = self._coerce_material_optics_def(item)
            snapshot[(entry.material_name, entry.light_type)] = entry
        return snapshot

    def _reaction_table_snapshot_payload(self) -> dict[str, object]:
        return {
            "actions": [asdict(action) for action in self.rulebook.reaction_actions[1:]],
            "rules": {
                "material_material": [asdict(rule) for rule in self.rulebook.material_material_rules],
                "material_gas": [asdict(rule) for rule in self.rulebook.material_gas_rules],
                "material_light": [asdict(rule) for rule in self.rulebook.material_light_rules],
                "gas_gas": [asdict(rule) for rule in self.rulebook.gas_gas_rules],
                "gas_light": [asdict(rule) for rule in self.rulebook.gas_light_rules],
                "self_rules": [asdict(rule) for rule in self.rulebook.self_rules],
            },
        }

    def _stable_shadow_payload(self, name: str, snapshot_factory: Callable[[], Any]) -> Any:
        payload = self.bridge.shadow_tables.get(str(name))
        if payload is not None:
            stable = deepcopy(payload)
            self._stable_shadow_payloads[str(name)] = stable
            return deepcopy(stable)
        cached = self._stable_shadow_payloads.get(str(name))
        if cached is not None:
            return deepcopy(cached)
        stable = deepcopy(snapshot_factory())
        self._stable_shadow_payloads[str(name)] = stable
        return deepcopy(stable)

    def _set_stable_shadow_payload(self, name: str, payload: Any) -> None:
        self._stable_shadow_payloads[str(name)] = deepcopy(payload)
    def _shadow_has_table_payload(self, name: str) -> bool:
        if self.bridge.shadow_tables.get(str(name)) is not None:
         return True
        return self._stable_shadow_payloads.get(str(name)) is not None

    def _merged_reaction_table_payload(
        self,
        actions: list[ReactionAction],
        rules: dict[str, list[object]],
    ) -> dict[str, object]:
        base_payload = self._shadow_reaction_payload()
        merged_actions = list(base_payload.get("actions", []))
        merged_actions.extend(asdict(action) for action in actions)
        merged_rules = {
            "material_material": list(base_payload.get("rules", {}).get("material_material", [])),
            "material_gas": list(base_payload.get("rules", {}).get("material_gas", [])),
            "material_light": list(base_payload.get("rules", {}).get("material_light", [])),
            "gas_gas": list(base_payload.get("rules", {}).get("gas_gas", [])),
            "gas_light": list(base_payload.get("rules", {}).get("gas_light", [])),
            "self_rules": list(base_payload.get("rules", {}).get("self_rules", [])),
        }
        for name, entries in rules.items():
            merged_rules[name].extend(asdict(rule) for rule in entries)
        return {
            "actions": merged_actions,
            "rules": merged_rules,
        }

    def _merged_material_table_payload(self, materials: list[MaterialDef]) -> list[dict[str, Any]]:
        base_payload = self._shadow_material_payload()
        merged = {int(item["material_id"]): dict(item) for item in base_payload}
        for material in materials:
            merged[int(material.material_id)] = asdict(material)
        return [merged[material_id] for material_id in sorted(merged)]

    def _merged_gas_species_table_payload(self, gases: list[GasSpeciesDef]) -> list[dict[str, Any]]:
        base_payload = self._shadow_gas_species_payload()
        merged = {int(item["species_id"]): dict(item) for item in base_payload}
        for gas in gases:
            merged[int(gas.species_id)] = asdict(gas)
        return [merged[species_id] for species_id in sorted(merged)]

    def _merged_light_type_table_payload(self, lights: list[LightTypeDef]) -> list[dict[str, Any]]:
        base_payload = self._shadow_light_type_payload()
        merged = {int(item["light_type_id"]): dict(item) for item in base_payload}
        for light in lights:
            merged[int(light.light_type_id)] = asdict(light)
        return [merged[light_id] for light_id in sorted(merged)]

    def _merged_material_optics_table_payload(self, optics: list[MaterialOpticsDef]) -> list[dict[str, Any]]:
        base_payload = self._stable_shadow_payload("optics", self._material_optics_table_snapshot_payload)
        merged = {(str(item["material_name"]), str(item["light_type"])): dict(item) for item in base_payload}
        for entry in optics:
            merged[(str(entry.material_name), str(entry.light_type))] = asdict(entry)
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def _coerce_enum(enum_type: type[Any], value: Any) -> Any:
        if isinstance(value, enum_type):
            return value
        if isinstance(value, str):
            if value in enum_type.__members__:
                return enum_type.__members__[value]
            return enum_type(value)
        return enum_type(value)

    def _coerce_material_def(self, material: MaterialDef | dict[str, Any]) -> MaterialDef:
        payload = asdict(material) if isinstance(material, MaterialDef) else dict(material)
        payload["default_phase"] = self._coerce_enum(Phase, payload["default_phase"])
        payload["base_color"] = tuple(payload["base_color"])
        payload["reaction_slots"] = tuple(payload.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
        payload["tags"] = tuple(payload.get("tags", ()))
        payload["collapse_generation"] = self._canonical_material_input_name(payload.get("collapse_generation"))
        payload["powder_generation"] = self._canonical_material_input_name(payload.get("powder_generation"))
        payload["melt_to_material"] = self._canonical_material_input_name(payload.get("melt_to_material"))
        payload["freeze_to_material"] = self._canonical_material_input_name(payload.get("freeze_to_material"))
        return MaterialDef(**payload)

    def _coerce_gas_species_def(self, gas: GasSpeciesDef | dict[str, Any]) -> GasSpeciesDef:
        payload = asdict(gas) if isinstance(gas, GasSpeciesDef) else dict(gas)
        payload["color"] = tuple(payload["color"])
        payload["condense_to_material"] = self._canonical_material_input_name(payload.get("condense_to_material"))
        return GasSpeciesDef(**payload)

    def _coerce_light_type_def(self, light: LightTypeDef | dict[str, Any]) -> LightTypeDef:
        if isinstance(light, LightTypeDef):
            return light
        payload = dict(light)
        payload["color"] = tuple(payload["color"])
        return LightTypeDef(**payload)

    @staticmethod
    def _canonical_material_input_name(name: str | None) -> str | None:
        if name is None:
            return None
        if name == "__random__":
            return name
        return BASE_MATERIAL_RUNTIME_ALIASES.get(str(name), str(name))

    def _coerce_material_optics_def(self, optics: MaterialOpticsDef | dict[str, Any]) -> MaterialOpticsDef:
        payload = asdict(optics) if isinstance(optics, MaterialOpticsDef) else dict(optics)
        payload["material_name"] = self._canonical_material_input_name(payload.get("material_name"))
        return MaterialOpticsDef(**payload)

    def _coerce_reaction_action(self, action: ReactionAction | dict[str, Any]) -> ReactionAction:
        payload = asdict(action) if isinstance(action, ReactionAction) else dict(action)
        payload["reaction_type"] = self._coerce_enum(ReactionType, payload["reaction_type"])
        payload["direction"] = self._coerce_enum(Direction, payload.get("direction", Direction.ALL))
        payload["velocity"] = tuple(payload.get("velocity", (0.0, 0.0)))
        payload["target_material"] = self._canonical_material_input_name(payload.get("target_material"))
        payload["emit_material"] = self._canonical_material_input_name(payload.get("emit_material"))
        return ReactionAction(**payload)

    def _coerce_pair_reaction_rule(self, rule: PairReactionRule | dict[str, Any]) -> PairReactionRule:
        payload = asdict(rule) if isinstance(rule, PairReactionRule) else dict(rule)
        payload["phases"] = tuple(self._coerce_enum(Phase, phase) for phase in payload.get("phases", ()))
        payload["lhs_material"] = self._canonical_material_input_name(payload.get("lhs_material"))
        payload["rhs_material"] = self._canonical_material_input_name(payload.get("rhs_material"))
        return PairReactionRule(**payload)

    def _coerce_self_reaction_rule(self, rule: SelfReactionRule | dict[str, Any]) -> SelfReactionRule:
        payload = asdict(rule) if isinstance(rule, SelfReactionRule) else dict(rule)
        payload["phases"] = tuple(self._coerce_enum(Phase, phase) for phase in payload.get("phases", ()))
        payload["material"] = self._canonical_material_input_name(payload.get("material"))
        return SelfReactionRule(**payload)

    def _coerce_reaction_rules(self, rules: dict[str, object]) -> dict[str, list[object]]:
        return {
            "material_material": [self._coerce_pair_reaction_rule(rule) for rule in rules.get("material_material", [])],
            "material_gas": [self._coerce_pair_reaction_rule(rule) for rule in rules.get("material_gas", [])],
            "material_light": [self._coerce_pair_reaction_rule(rule) for rule in rules.get("material_light", [])],
            "gas_gas": [self._coerce_pair_reaction_rule(rule) for rule in rules.get("gas_gas", [])],
            "gas_light": [self._coerce_pair_reaction_rule(rule) for rule in rules.get("gas_light", [])],
            "self_rules": [self._coerce_self_reaction_rule(rule) for rule in rules.get("self_rules", [])],
        }

    def _shadow_material_payload(self) -> list[dict[str, Any]]:
        return self._stable_shadow_payload("materials", self._material_table_snapshot_payload)

    def _shadow_gas_species_payload(self) -> list[dict[str, Any]]:
        return self._stable_shadow_payload("gases", self._gas_species_table_snapshot_payload)

    def _shadow_light_type_payload(self) -> list[dict[str, Any]]:
        return self._stable_shadow_payload("lights", self._light_type_table_snapshot_payload)

    def _shadow_reaction_payload(self) -> dict[str, Any]:
        return self._stable_shadow_payload("reactions", self._reaction_table_snapshot_payload)

    @staticmethod
    def _payload_name_set(payload: Iterable[dict[str, Any]], field: str = "name") -> set[str]:
        names: set[str] = set()
        for item in payload:
            name = item.get(field)
            if name:
                names.add(str(name))
        return names

    @staticmethod
    def _validate_named_reference(valid_names: set[str], reference: str | None) -> None:
        if reference is None:
            return
        if str(reference) not in valid_names:
            raise KeyError(reference)

    @staticmethod
    def _validate_unique_identity_fields(
        payload: Iterable[dict[str, Any]],
        *,
        id_field: str,
        name_field: str = "name",
        allow_zero_id: bool,
    ) -> None:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        for item in payload:
            item_id = int(item[id_field])
            item_name = str(item[name_field])
            if not allow_zero_id and item_id == 0:
                raise ValueError(f"{id_field}=0 is reserved")
            if item_id in seen_ids:
                raise ValueError(f"duplicate {id_field}: {item_id}")
            if item_name in seen_names:
                raise ValueError(f"duplicate {name_field}: {item_name}")
            seen_ids.add(item_id)
            seen_names.add(item_name)

    def _validate_material_table_payload(self, materials_payload: list[dict[str, Any]]) -> None:
        self._validate_unique_identity_fields(
            materials_payload,
            id_field="material_id",
            allow_zero_id=False,
        )
        material_names = self._payload_name_set(materials_payload)
        gas_names = self._payload_name_set(self._shadow_gas_species_payload())
        reactions_payload = self.bridge.shadow_tables.get("reactions")
        action_count = 0
        if reactions_payload is not None:
            action_count = len(reactions_payload.get("actions", [])) + 1
        for item in materials_payload:
            item_name = str(item.get("name", ""))
            reaction_slots = tuple(int(slot) for slot in item.get("reaction_slots", (-1,) * 8))
            if len(reaction_slots) != 8:
                raise ValueError("reaction_slots must contain exactly 8 entries")
            if action_count > 0:
                for action_index in reaction_slots:
                    if action_index < -1 or action_index >= action_count:
                        raise IndexError(action_index)
            is_placeholder = str(item.get("render_group", "")) == "placeholder" or "placeholder" in tuple(
                str(tag) for tag in item.get("tags", ())
            )
            if item_name == "placeholder_solid" or is_placeholder:
                if bool(item.get("is_structural", False)):
                    raise ValueError("placeholder materials cannot be structural")
                if bool(item.get("is_support_anchor", False)):
                    raise ValueError("placeholder materials cannot be support anchors")
            for field in ("collapse_generation", "powder_generation", "melt_to_material", "freeze_to_material"):
                self._validate_named_reference(material_names, item.get(field))
            if gas_names:
                self._validate_named_reference(gas_names, item.get("boil_to_gas_species"))
        if reactions_payload is not None:
            uses_random_convert = any(
                action.get("target_material") == "__random__"
                for action in reactions_payload.get("actions", [])
            )
            if uses_random_convert:
                chaos_convert_candidates = [
                    item
                    for item in materials_payload
                    if "chaos_convert" in tuple(str(tag) for tag in item.get("tags", ()))
                    and int(self._coerce_enum(Phase, item.get("default_phase", Phase.STATIC_SOLID))) == int(Phase.POWDER)
                ]
                if not chaos_convert_candidates:
                    raise ValueError(
                        "material table must contain at least one powder material tagged chaos_convert when reactions use target_material=__random__"
                    )

    def _validate_gas_species_payload(self, gases_payload: list[dict[str, Any]]) -> None:
        air_entries = [item for item in gases_payload if str(item.get("name", "")) == "air"]
        if len(air_entries) != 1:
            raise ValueError("gas table must contain exactly one air species")
        air_entry = air_entries[0]
        if int(air_entry["species_id"]) != 0:
            raise ValueError("air species_id must remain 0")
        if air_entry.get("condense_to_material") is not None:
            raise ValueError("air cannot condense to a material")
        self._validate_unique_identity_fields(
            gases_payload,
            id_field="species_id",
            allow_zero_id=True,
        )
        material_names = self._payload_name_set(self._shadow_material_payload())
        for item in gases_payload:
            self._validate_named_reference(material_names, item.get("condense_to_material"))

    def _validate_light_type_payload(self, lights_payload: list[dict[str, Any]]) -> None:
        self._validate_unique_identity_fields(
            lights_payload,
            id_field="light_type_id",
            allow_zero_id=True,
        )
        if len(lights_payload) > 8:
            raise ValueError("light type count exceeds maximum of 8")
        seen_dose_channels: set[int] = set()
        for item in lights_payload:
            light_type_id = int(item["light_type_id"])
            if light_type_id < 0 or light_type_id >= 8:
                raise ValueError(f"light_type_id out of range: {light_type_id}")
            dose_channel_id = int(item.get("dose_channel_id", -1))
            if dose_channel_id < 0 or dose_channel_id >= 8:
                raise ValueError(f"dose_channel_id out of range: {dose_channel_id}")
            if dose_channel_id in seen_dose_channels:
                raise ValueError(f"duplicate dose_channel_id: {dose_channel_id}")
            seen_dose_channels.add(dose_channel_id)

    def _validate_material_optics_payload(self, optics_payload: list[dict[str, Any]]) -> None:
        material_names = self._payload_name_set(self._shadow_material_payload())
        light_names = self._payload_name_set(self._shadow_light_type_payload())
        for item in optics_payload:
            self._validate_named_reference(material_names, item.get("material_name"))
            self._validate_named_reference(light_names, item.get("light_type"))

    def _validate_reaction_payload(self, reactions_payload: dict[str, Any]) -> None:
        material_names = self._payload_name_set(self._shadow_material_payload())
        gas_names = self._payload_name_set(self._shadow_gas_species_payload())
        light_names = self._payload_name_set(self._shadow_light_type_payload())
        valid_consume_policies = {"none", "lhs", "rhs", "both"}
        actions_payload = list(reactions_payload.get("actions", []))
        action_count = len(actions_payload) + 1
        for action in actions_payload:
            reaction_type = self._coerce_enum(ReactionType, action.get("reaction_type", ReactionType.NONE))
            if reaction_type == ReactionType.NONE:
                raise ValueError("reaction action 0 is reserved for ReactionType.NONE")
            duration = int(action.get("duration", 0))
            if duration < 0:
                raise ValueError(f"reaction actions require non-negative duration: {duration}")
            generation = int(action.get("generation", 0))
            if generation != 0:
                raise ValueError("reaction actions do not support non-zero generation")
            if bool(action.get("allow_subunit_scale", False)) and reaction_type != ReactionType.CONVERT_MATERIAL:
                raise ValueError("allow_subunit_scale is only supported for convert_material actions")
            target_material = action.get("target_material")
            if target_material == "__random__" and reaction_type != ReactionType.CONVERT_MATERIAL:
                raise ValueError("target_material=__random__ is only supported for convert_material actions")
            if target_material is not None and reaction_type != ReactionType.CONVERT_MATERIAL:
                if reaction_type != ReactionType.HARM:
                    raise ValueError("target_material is only supported for convert_material and harm actions")
            if action.get("emit_material") is not None and reaction_type != ReactionType.EMIT_MATERIAL:
                raise ValueError("emit_material is only supported for emit_material actions")
            if action.get("light_type") is not None and reaction_type != ReactionType.EMIT_LIGHT:
                raise ValueError("light_type is only supported for emit_light actions")
            if action.get("gas_species") is not None and reaction_type != ReactionType.MODIFY_GAS:
                raise ValueError("gas_species is only supported for modify_gas actions")
            if float(action.get("delta", 0.0)) != 0.0 and reaction_type != ReactionType.MODIFY_TEMPERATURE:
                raise ValueError("delta is only supported for modify_temperature actions")
            if float(action.get("value", 0.0)) != 0.0 and reaction_type != ReactionType.HARM:
                raise ValueError("value is only supported for harm actions")
            velocity_value = action.get("velocity", (0.0, 0.0))
            velocity_xy = tuple(float(component) for component in velocity_value)
            if any(abs(component) > 1.0e-6 for component in velocity_xy) and reaction_type != ReactionType.EMIT_MATERIAL:
                raise ValueError("velocity is only supported for emit_material actions")
            if float(action.get("beam_width", 1.0)) != 1.0 and reaction_type != ReactionType.EMIT_LIGHT:
                raise ValueError("beam_width is only supported for emit_light actions")
            if target_material != "__random__":
                self._validate_named_reference(material_names, target_material)
            self._validate_named_reference(material_names, action.get("emit_material"))
            self._validate_named_reference(gas_names, action.get("gas_species"))
            self._validate_named_reference(light_names, action.get("light_type"))
            if reaction_type == ReactionType.EMIT_MATERIAL and action.get("emit_material") is None:
                raise ValueError("emit_material actions require emit_material")
            if reaction_type == ReactionType.EMIT_MATERIAL and float(action.get("speed", 0.0)) < 0.0:
                raise ValueError("emit_material actions require non-negative speed")
            if reaction_type == ReactionType.EMIT_LIGHT and action.get("light_type") is None:
                raise ValueError("emit_light actions require light_type")
            if reaction_type == ReactionType.EMIT_LIGHT and float(action.get("strength", 0.0)) <= 0.0:
                raise ValueError("emit_light actions require positive strength")
            if reaction_type == ReactionType.EMIT_LIGHT and float(action.get("beam_width", 1.0)) < 0.0:
                raise ValueError("emit_light actions require non-negative beam_width")
            if reaction_type == ReactionType.EMIT_LIGHT and int(action.get("range_cells", 0)) < 0:
                raise ValueError("emit_light actions require non-negative range_cells")
            if reaction_type == ReactionType.MODIFY_GAS and action.get("gas_species") is None:
                raise ValueError("modify_gas actions require gas_species")
            if reaction_type == ReactionType.CONVERT_MATERIAL and float(action.get("harm_per_frame", 0.0)) < 0.0:
                raise ValueError("convert_material actions require non-negative harm_per_frame")
        rules_payload = dict(reactions_payload.get("rules", {}))
        for rule_set in PAIR_REACTION_RULE_SET_NAMES:
            for rule in rules_payload.get(rule_set, []):
                self._validate_named_reference(material_names, rule.get("lhs_material"))
                self._validate_named_reference(material_names, rule.get("rhs_material"))
                self._validate_named_reference(gas_names, rule.get("lhs_gas"))
                self._validate_named_reference(gas_names, rule.get("rhs_gas"))
                self._validate_named_reference(light_names, rule.get("rhs_light"))
                trigger_slot_value = rule.get("trigger_slot_index", -1)
                trigger_slot_index = -1 if trigger_slot_value is None else int(trigger_slot_value)
                if trigger_slot_index < -1 or trigger_slot_index >= 8:
                    raise IndexError(trigger_slot_index)
                result_action = int(rule.get("result_action", 0))
                if result_action < -1 or result_action >= action_count:
                    raise IndexError(result_action)
                min_temperature = float(rule.get("min_temperature", float("-inf")))
                max_temperature = float(rule.get("max_temperature", float("inf")))
                if min_temperature > max_temperature:
                    raise ValueError(f"{rule_set} rule requires min_temperature <= max_temperature")
                threshold = float(rule.get("threshold", 0.0))
                if threshold < 0.0:
                    raise ValueError(f"{rule_set} rule requires non-negative threshold")
                rate = float(rule.get("rate", 1.0))
                if rate < 0.0:
                    raise ValueError(f"{rule_set} rule requires non-negative rate")
                has_trigger_slot = trigger_slot_index >= 0
                has_result_action = result_action > 0
                consume_policy = str(rule.get("consume_policy", "none") or "none").lower()
                if consume_policy not in valid_consume_policies:
                    raise ValueError(f"invalid consume_policy: {consume_policy}")
                if has_trigger_slot and has_result_action:
                    raise ValueError(f"{rule_set} rule cannot define both trigger_slot_index and result_action")
                if rule_set in {"gas_gas", "gas_light"} and has_trigger_slot:
                    raise ValueError(f"{rule_set} rules cannot use trigger_slot_index")
        for rule in rules_payload.get("self_rules", []):
            self._validate_named_reference(material_names, rule.get("material"))
            trigger_slot_value = rule.get("trigger_slot_index", -1)
            trigger_slot_index = -1 if trigger_slot_value is None else int(trigger_slot_value)
            if trigger_slot_index < 0 or trigger_slot_index >= 8:
                raise IndexError(trigger_slot_index)
            timer_index_value = rule.get("timer_index", None)
            timer_index = -1 if timer_index_value is None else int(timer_index_value)
            if timer_index < -1:
                raise IndexError(timer_index)
            if timer_index >= 4:
                raise IndexError(timer_index)
            if timer_index >= 0:
                if trigger_slot_index >= 4:
                    raise ValueError("untimed self rules cannot define timer_index")
                if timer_index != trigger_slot_index:
                    raise ValueError("self rule timer_index must match trigger_slot_index for timed slots")
            min_temperature = float(rule.get("min_temperature", float("-inf")))
            max_temperature = float(rule.get("max_temperature", float("inf")))
            if min_temperature > max_temperature:
                raise ValueError("self rules require min_temperature <= max_temperature")
            integrity_at_most = rule.get("integrity_at_most", None)
            integrity_at_least = rule.get("integrity_at_least", None)
            if integrity_at_most is not None and integrity_at_least is not None:
                if float(integrity_at_most) < float(integrity_at_least):
                    raise ValueError("self rules require integrity_at_most >= integrity_at_least")

    def _normalize_material_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_json_payload_value(
                str(self._canonical_material_input_name(value))
                if key in {"collapse_generation", "powder_generation", "melt_to_material", "freeze_to_material"}
                else value
            )
            for key, value in fields.items()
        }

    def _normalize_gas_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_json_payload_value(
                str(self._canonical_material_input_name(value)) if key == "condense_to_material" else value
            )
            for key, value in fields.items()
        }

    def _normalize_material_optics_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_json_payload_value(
                str(self._canonical_material_input_name(value)) if key == "material_name" else value
            )
            for key, value in fields.items()
        }

    def _normalize_reaction_action_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_json_payload_value(
                str(self._canonical_material_input_name(value))
                if key in {"target_material", "emit_material"}
                else value
            )
            for key, value in fields.items()
        }

    def _normalize_reaction_rule_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_json_payload_value(
                str(self._canonical_material_input_name(value))
                if key in {"lhs_material", "rhs_material", "material"}
                else value
            )
            for key, value in fields.items()
        }

    def _coerce_force_source(self, force_source: ForceSource | dict[str, Any]) -> ForceSource:
        if isinstance(force_source, ForceSource):
            return force_source
        payload = dict(force_source)
        payload["direction"] = tuple(payload.get("direction", (0.0, 0.0)))
        return ForceSource(**payload)

    def _public_force_source_input(self, force_source: ForceSource | dict[str, Any]) -> ForceSource:
        force_source = self._coerce_force_source(force_source)
        world_x = float(force_source.x) if force_source.world_x is None else float(force_source.world_x)
        world_y = float(force_source.y) if force_source.world_y is None else float(force_source.world_y)
        return replace(force_source, world_x=world_x, world_y=world_y)

    def _frame_force_source_input(self, force_source: ForceSource | dict[str, Any]) -> ForceSource:
        force_source = self._coerce_force_source(force_source)
        if force_source.world_x is not None and force_source.world_y is not None:
            return replace(force_source, world_x=float(force_source.world_x), world_y=float(force_source.world_y))
        world_x, world_y = self._buffer_to_world_float_position((float(force_source.x), float(force_source.y)))
        return replace(force_source, world_x=float(world_x), world_y=float(world_y))

    def _coerce_emitter(self, emitter: dict[str, Any]) -> dict[str, object]:
        payload = dict(emitter)
        light_id = self._resolve_sanctioned_light_id(str(payload["light_type"]))
        if light_id < 0:
            raise KeyError(payload["light_type"])
        x = int(payload["x"]) if "x" in payload else int(payload["origin"][0])
        y = int(payload["y"]) if "y" in payload else int(payload["origin"][1])
        radius = payload.get("radius", payload.get("range_cells"))
        if radius is None:
            resolved_range = self._shadow_light_default_range(light_id)
            if resolved_range is None:
                raise KeyError(payload["light_type"])
            radius = int(resolved_range)
        direction = tuple(float(value) for value in payload.get("direction", (0.0, 0.0)))
        shadow_light = self._shadow_light_name(light_id)
        if shadow_light is None:
            raise KeyError(payload["light_type"])
        return {
            "light_type": shadow_light,
            "origin": (x, y),
            "world_origin": (
                (int(payload["world_origin"][0]), int(payload["world_origin"][1]))
                if "world_origin" in payload
                else (x, y)
            ),
            "direction": direction,
            "spread": float(payload.get("spread", 0.25)),
            "strength": float(payload.get("strength", 1.0)),
            "range_cells": int(radius),
        }

    def _frame_emitter_input(self, emitter: dict[str, Any]) -> dict[str, object]:
        record = self._coerce_emitter(emitter)
        if "world_origin" in emitter:
            return {
                **dict(record),
                "origin": self._world_to_buffer_clamped(
                    int(record["world_origin"][0]),
                    int(record["world_origin"][1]),
                ),
            }
        origin = (int(record["origin"][0]), int(record["origin"][1]))
        return {
            **dict(record),
            "origin": origin,
            "world_origin": self._buffer_to_world_position(origin),
        }

    def _coerce_entity_placeholder(self, placeholder: EntityPlaceholder | dict[str, Any]) -> EntityPlaceholder:
        if isinstance(placeholder, EntityPlaceholder):
            return replace(
                placeholder,
                entity_id=int(placeholder.entity_id),
                x=int(placeholder.x),
                y=int(placeholder.y),
                width=int(placeholder.width),
                height=int(placeholder.height),
                material=str(self._canonical_material_input_name(placeholder.material)),
                world_x=None if placeholder.world_x is None else int(placeholder.world_x),
                world_y=None if placeholder.world_y is None else int(placeholder.world_y),
            )
        payload = dict(placeholder)
        return EntityPlaceholder(
            entity_id=int(payload["entity_id"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            material=str(self._canonical_material_input_name(payload.get("material", "placeholder_solid"))),
            world_x=None if payload.get("world_x") is None else int(payload["world_x"]),
            world_y=None if payload.get("world_y") is None else int(payload["world_y"]),
        )

    def _public_entity_placeholder_input(
        self,
        placeholder: EntityPlaceholder | dict[str, Any],
    ) -> EntityPlaceholder:
        placeholder = self._coerce_entity_placeholder(placeholder)
        if placeholder.world_x is not None and placeholder.world_y is not None:
            world_x = int(placeholder.world_x)
            world_y = int(placeholder.world_y)
        elif 0 <= int(placeholder.x) < self.width and 0 <= int(placeholder.y) < self.height:
            world_x, world_y = self._buffer_to_world_position((int(placeholder.x), int(placeholder.y)))
        else:
            world_x = int(placeholder.x)
            world_y = int(placeholder.y)
        return replace(placeholder, world_x=world_x, world_y=world_y)

    def _frame_entity_placeholder_input(
        self,
        placeholder: EntityPlaceholder | dict[str, Any],
    ) -> EntityPlaceholder:
        placeholder = self._coerce_entity_placeholder(placeholder)
        if placeholder.world_x is not None and placeholder.world_y is not None:
            buffer_x, buffer_y = self._world_to_buffer_clamped(int(placeholder.world_x), int(placeholder.world_y))
            world_x = int(placeholder.world_x)
            world_y = int(placeholder.world_y)
        else:
            world_x = int(placeholder.x)
            world_y = int(placeholder.y)
            buffer_x, buffer_y = self._world_to_buffer_clamped(world_x, world_y)
        return replace(
            placeholder,
            x=int(buffer_x),
            y=int(buffer_y),
            world_x=int(world_x),
            world_y=int(world_y),
        )

    def _coerce_entity_state(self, entity: EntityState | dict[str, Any]) -> EntityState:
        if isinstance(entity, EntityState):
            return replace(
                entity,
                entity_id=int(entity.entity_id),
                x=int(entity.x),
                y=int(entity.y),
                width=int(entity.width),
                height=int(entity.height),
                velocity_xy=(float(entity.velocity_xy[0]), float(entity.velocity_xy[1])),
                facing_xy=None if entity.facing_xy is None else (float(entity.facing_xy[0]), float(entity.facing_xy[1])),
                placeholder_material=str(self._canonical_material_input_name(entity.placeholder_material)),
                tags=tuple(str(item) for item in entity.tags),
                observe_channels=self._normalize_readback_channels(entity.observe_channels),
                observe_pad_cells=int(entity.observe_pad_cells),
                observe_width=None if entity.observe_width is None else int(entity.observe_width),
                observe_height=None if entity.observe_height is None else int(entity.observe_height),
                observe_label=None if entity.observe_label is None else str(entity.observe_label),
                world_x=None if entity.world_x is None else int(entity.world_x),
                world_y=None if entity.world_y is None else int(entity.world_y),
            )
        payload = dict(entity)
        payload["velocity_xy"] = tuple(payload.get("velocity_xy", (0.0, 0.0)))
        payload["facing_xy"] = None if payload.get("facing_xy") is None else tuple(payload["facing_xy"])
        payload["placeholder_material"] = str(
            self._canonical_material_input_name(payload.get("placeholder_material", "placeholder_solid"))
        )
        payload["tags"] = tuple(payload.get("tags", ()))
        payload["observe_channels"] = self._normalize_readback_channels(payload.get("observe_channels", ()))
        payload["world_x"] = None if payload.get("world_x") is None else int(payload["world_x"])
        payload["world_y"] = None if payload.get("world_y") is None else int(payload["world_y"])
        return EntityState(**payload)

    def _public_entity_state_input(self, entity: EntityState | dict[str, Any]) -> EntityState:
        entity = self._coerce_entity_state(entity)
        if entity.world_x is not None and entity.world_y is not None:
            world_x = int(entity.world_x)
            world_y = int(entity.world_y)
        elif 0 <= int(entity.x) < self.width and 0 <= int(entity.y) < self.height:
            world_x, world_y = self._buffer_to_world_position((int(entity.x), int(entity.y)))
        else:
            world_x = int(entity.x)
            world_y = int(entity.y)
        return replace(entity, world_x=world_x, world_y=world_y)

    def _frame_entity_state_input(self, entity: EntityState | dict[str, Any]) -> EntityState:
        entity = self._coerce_entity_state(entity)
        if entity.world_x is not None and entity.world_y is not None:
            buffer_x, buffer_y = self._world_to_buffer_clamped(int(entity.world_x), int(entity.world_y))
            world_x = int(entity.world_x)
            world_y = int(entity.world_y)
        else:
            buffer_x = int(entity.x)
            buffer_y = int(entity.y)
            world_x, world_y = self._buffer_to_world_position((buffer_x, buffer_y))
        return replace(
            entity,
            x=int(buffer_x),
            y=int(buffer_y),
            world_x=int(world_x),
            world_y=int(world_y),
        )

    def _coerce_entity_observation_spec(
        self,
        spec: EntityObservationSpec | dict[str, Any],
    ) -> EntityObservationSpec:
        if isinstance(spec, EntityObservationSpec):
            return replace(
                spec,
                entity_id=int(spec.entity_id),
                observe_channels=self._normalize_readback_channels(spec.observe_channels),
                observe_pad_cells=int(spec.observe_pad_cells),
                observe_width=None if spec.observe_width is None else int(spec.observe_width),
                observe_height=None if spec.observe_height is None else int(spec.observe_height),
                observe_label=None if spec.observe_label is None else str(spec.observe_label),
            )
        payload = dict(spec)
        return EntityObservationSpec(
            entity_id=int(payload["entity_id"]),
            observe_channels=self._normalize_readback_channels(payload.get("observe_channels", ())),
            observe_pad_cells=int(payload.get("observe_pad_cells", 0)),
            observe_width=None if payload.get("observe_width") is None else int(payload["observe_width"]),
            observe_height=None if payload.get("observe_height") is None else int(payload["observe_height"]),
            observe_label=None if payload.get("observe_label") is None else str(payload["observe_label"]),
        )

    def _normalize_entity_state_patch_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for name, value in fields.items():
            if name not in ENTITY_STATE_PATCHABLE_FIELDS and name not in ENTITY_STATE_PATCH_METADATA_FIELDS:
                raise KeyError(name)
            if name in {"x", "y", "width", "height", "observe_pad_cells", "_world_x", "_world_y"}:
                normalized[name] = int(value)
            elif name == "velocity_xy":
                normalized[name] = tuple(float(item) for item in value)
            elif name == "facing_xy":
                normalized[name] = None if value is None else tuple(float(item) for item in value)
            elif name == "placeholder_material":
                normalized[name] = str(self._canonical_material_input_name(value))
            elif name == "tags":
                normalized[name] = tuple(str(item) for item in value)
            elif name == "observe_channels":
                normalized[name] = self._normalize_readback_channels(value)
            elif name in {"observe_width", "observe_height"}:
                normalized[name] = None if value is None else int(value)
            elif name == "observe_label":
                normalized[name] = None if value is None else str(value)
        return normalized

    def _public_entity_state_patch_input(
        self,
        patch: EntityStatePatch | dict[str, Any],
    ) -> EntityStatePatch:
        patch = self._coerce_entity_state_patch(patch)
        fields = dict(patch.fields)
        if "x" in fields and "y" in fields:
            if 0 <= int(fields["x"]) < self.width and 0 <= int(fields["y"]) < self.height:
                world_x, world_y = self._buffer_to_world_position((int(fields["x"]), int(fields["y"])))
                fields["_world_x"] = int(world_x)
                fields["_world_y"] = int(world_y)
            else:
                fields["_world_x"] = int(fields["x"])
                fields["_world_y"] = int(fields["y"])
        else:
            if "x" in fields:
                fields["_world_x"] = int(fields["x"])
            if "y" in fields:
                fields["_world_y"] = int(fields["y"])
        return EntityStatePatch(entity_id=int(patch.entity_id), fields=self._normalize_entity_state_patch_fields(fields))

    def _controller_turn_entity_input(self, entity: EntityState | dict[str, Any]) -> EntityState:
        entity = self._coerce_entity_state(entity)
        return replace(
            entity,
            world_x=None if entity.world_x is None else int(entity.world_x),
            world_y=None if entity.world_y is None else int(entity.world_y),
        )

    def _frame_entity_state_patch_input(
        self,
        patch: EntityStatePatch | dict[str, Any],
    ) -> EntityStatePatch:
        patch = self._coerce_entity_state_patch(patch)
        fields = dict(patch.fields)
        if "_world_x" in fields:
            buffer_x, _ = self._world_to_buffer_clamped(int(fields["_world_x"]), int(fields.get("_world_y", 0)))
            fields["x"] = int(buffer_x)
        if "_world_y" in fields:
            _, buffer_y = self._world_to_buffer_clamped(int(fields.get("_world_x", 0)), int(fields["_world_y"]))
            fields["y"] = int(buffer_y)
        return EntityStatePatch(entity_id=int(patch.entity_id), fields=self._normalize_entity_state_patch_fields(fields))

    def _coerce_entity_state_patch(self, patch: EntityStatePatch | dict[str, Any]) -> EntityStatePatch:
        if isinstance(patch, EntityStatePatch):
            return EntityStatePatch(
                entity_id=int(patch.entity_id),
                fields=self._normalize_entity_state_patch_fields(dict(patch.fields)),
            )
        payload = dict(patch)
        return EntityStatePatch(
            entity_id=int(payload["entity_id"]),
            fields=self._normalize_entity_state_patch_fields(dict(payload.get("fields", {}))),
        )

    def _coerce_observation_target(self, target: ObservationTarget | dict[str, Any]) -> ObservationTarget:
        if isinstance(target, ObservationTarget):
            return target
        payload = dict(target)
        return ObservationTarget(
            observer_id=int(payload["observer_id"]),
            channels=tuple(payload.get("channels", ())),
            center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
            center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
            width=None if payload.get("width") is None else int(payload["width"]),
            height=None if payload.get("height") is None else int(payload["height"]),
            entity_id=None if payload.get("entity_id") is None else int(payload["entity_id"]),
            pad_cells=int(payload.get("pad_cells", 0)),
            label=None if payload.get("label") is None else str(payload["label"]),
            target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
            target_dx=int(payload.get("target_dx", 0)),
            target_dy=int(payload.get("target_dy", 0)),
        )

    def _coerce_target_query(self, query: TargetQuery | dict[str, Any]) -> TargetQuery:
        if isinstance(query, TargetQuery):
            return query
        payload = dict(query)
        return TargetQuery(
            query_id=str(payload["query_id"]),
            anchor_filters=tuple(str(item) for item in payload.get("anchor_filters", ())),
            source_entity_id=None if payload.get("source_entity_id") is None else int(payload["source_entity_id"]),
            source_x=None if payload.get("source_x") is None else int(payload["source_x"]),
            source_y=None if payload.get("source_y") is None else int(payload["source_y"]),
            anchor_entity_id=None if payload.get("anchor_entity_id") is None else int(payload["anchor_entity_id"]),
            direction=None if payload.get("direction") is None else str(payload["direction"]),
            distance_cells=int(payload.get("distance_cells", 0)),
            distance_meters=None if payload.get("distance_meters") is None else float(payload["distance_meters"]),
            distance_hint=None if payload.get("distance_hint") is None else str(payload["distance_hint"]),
            require_empty=bool(payload.get("require_empty", False)),
            search_radius=int(payload.get("search_radius", 0)),
            label=None if payload.get("label") is None else str(payload["label"]),
        )

    def _coerce_change_intent(self, intent: ChangeIntent | dict[str, Any]) -> ChangeIntent:
        if isinstance(intent, ChangeIntent):
            return intent
        payload = dict(intent)
        return ChangeIntent(
            intent_id=str(payload["intent_id"]),
            target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
            center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
            center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
            target_dx=int(payload.get("target_dx", 0)),
            target_dy=int(payload.get("target_dy", 0)),
            radius=int(payload.get("radius", 0)),
            material=None if payload.get("material") is None else str(self._canonical_material_input_name(payload["material"])),
            temperature_delta=float(payload.get("temperature_delta", 0.0)),
            velocity=None if payload.get("velocity") is None else tuple(float(value) for value in payload["velocity"]),
            velocity_carrier=str(payload.get("velocity_carrier", "cell")),
            velocity_mode=str(payload.get("velocity_mode", "add")),
            require_empty=bool(payload.get("require_empty", False)),
            fallback_mode=str(payload.get("fallback_mode", "nearest_empty")),
            fallback_radius=int(payload.get("fallback_radius", 0)),
            potency=float(payload.get("potency", 1.0)),
            stability=float(payload.get("stability", 1.0)),
            label=None if payload.get("label") is None else str(payload["label"]),
        )

    def _coerce_carrier_intent(self, intent: CarrierIntent | dict[str, Any]) -> CarrierIntent:
        if isinstance(intent, CarrierIntent):
            return intent
        payload = dict(intent)
        return CarrierIntent(
            intent_id=str(payload["intent_id"]),
            kind=str(payload["kind"]),
            target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
            center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
            center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
            source_entity_id=None if payload.get("source_entity_id") is None else int(payload["source_entity_id"]),
            source_x=None if payload.get("source_x") is None else int(payload["source_x"]),
            source_y=None if payload.get("source_y") is None else int(payload["source_y"]),
            target_dx=int(payload.get("target_dx", 0)),
            target_dy=int(payload.get("target_dy", 0)),
            radius=int(payload.get("radius", 0)),
            material=None if payload.get("material") is None else str(self._canonical_material_input_name(payload["material"])),
            gas_species=None if payload.get("gas_species") is None else str(payload["gas_species"]),
            gas_amount=float(payload.get("gas_amount", 0.0)),
            light_type=None if payload.get("light_type") is None else str(payload["light_type"]),
            light_strength=float(payload.get("light_strength", 1.0)),
            light_spread=float(payload.get("light_spread", 0.25)),
            force_radius=float(payload.get("force_radius", 0.0)),
            force_strength=float(payload.get("force_strength", 0.0)),
            force_lifetime=float(payload.get("force_lifetime", 0.5)),
            release_mode=str(payload.get("release_mode", "impact")),
            require_empty=bool(payload.get("require_empty", False)),
            fallback_mode=str(payload.get("fallback_mode", "nearest_empty")),
            fallback_radius=int(payload.get("fallback_radius", 0)),
            potency=float(payload.get("potency", 1.0)),
            stability=float(payload.get("stability", 1.0)),
            label=None if payload.get("label") is None else str(payload["label"]),
        )

    def _coerce_readback_request(self, request: ReadbackRequest | dict[str, Any]) -> ReadbackRequest:
        if isinstance(request, ReadbackRequest):
            return self._normalize_readback_request(replace(request))
        payload = dict(request)
        return self._normalize_readback_request(
            ReadbackRequest(
                request_id=None if payload.get("request_id") is None else int(payload["request_id"]),
                center_x=None if payload.get("center_x") is None else int(payload["center_x"]),
                center_y=None if payload.get("center_y") is None else int(payload["center_y"]),
                width=int(payload.get("width", 1)),
                height=int(payload.get("height", 1)),
                channels=tuple(payload.get("channels", ())),
                observer_id=None if payload.get("observer_id") is None else int(payload["observer_id"]),
                label=None if payload.get("label") is None else str(payload["label"]),
                target_query_id=None if payload.get("target_query_id") is None else str(payload["target_query_id"]),
                target_dx=int(payload.get("target_dx", 0)),
                target_dy=int(payload.get("target_dy", 0)),
            )
        )

    @staticmethod
    def _normalize_readback_channels(channels: Any) -> tuple[str, ...]:
        return tuple(
            channel
            for channel in dict.fromkeys(str(channel) for channel in channels)
            if channel in READBACK_ALLOWED_CHANNEL_SET
        )

    def _normalize_readback_request(self, request: ReadbackRequest) -> ReadbackRequest:
        width = max(1, min(MAX_ASYNC_READBACK_WIDTH, int(request.width)))
        height = max(1, min(MAX_ASYNC_READBACK_HEIGHT, int(request.height)))
        channels = self._normalize_readback_channels(request.channels)
        return ReadbackRequest(
            request_id=None if request.request_id is None else int(request.request_id),
            center_x=None if request.center_x is None else int(request.center_x),
            center_y=None if request.center_y is None else int(request.center_y),
            width=width,
            height=height,
            channels=channels,
            observer_id=request.observer_id,
            label=request.label,
            target_query_id=request.target_query_id,
            target_dx=int(request.target_dx),
            target_dy=int(request.target_dy),
        )

    def _assign_readback_request_id(self, request: ReadbackRequest) -> ReadbackRequest:
        if request.request_id is None:
            request_id = self.next_readback_request_id
            self.next_readback_request_id += 1
            self.canceled_readback_request_ids.discard(int(request_id))
            return replace(request, request_id=int(request_id))
        request_id = int(request.request_id)
        self.next_readback_request_id = max(self.next_readback_request_id, request_id + 1)
        self.canceled_readback_request_ids.discard(request_id)
        return replace(request, request_id=request_id)

    def _assign_preview_readback_request_ids(
        self,
        requests: list[ReadbackRequest],
        *,
        next_request_id: int | None = None,
    ) -> tuple[list[ReadbackRequest], int]:
        predicted_next = int(self.next_readback_request_id if next_request_id is None else next_request_id)
        assigned_requests: list[ReadbackRequest] = []
        for request in requests:
            normalized_request = self._normalize_readback_request(request)
            if normalized_request.request_id is None:
                normalized_request = replace(normalized_request, request_id=int(predicted_next))
                predicted_next += 1
            else:
                request_id = int(normalized_request.request_id)
                normalized_request = replace(normalized_request, request_id=request_id)
                predicted_next = max(predicted_next, request_id + 1)
            assigned_requests.append(normalized_request)
        return assigned_requests, predicted_next

    def _coerce_world_command(self, command: WorldCommand | dict[str, Any]) -> WorldCommand:
        if isinstance(command, WorldCommand):
            kind = str(command.kind)
            payload = deepcopy(command.payload)
        else:
            raw = dict(command)
            kind = str(raw["kind"])
            payload = deepcopy(dict(raw.get("payload", {})))
        if kind in {"inject_material", "write_material_region"} and payload.get("material") is not None:
            payload["material"] = str(self._canonical_material_input_name(payload["material"]))
        elif kind == "sync_entity_states" and isinstance(payload.get("entities"), list):
            payload["entities"] = [
                asdict(self._public_entity_state_input(entity))
                for entity in payload.get("entities", [])
            ]
        elif kind == "sync_entity_placeholders" and isinstance(payload.get("placeholders"), list):
            payload["placeholders"] = [
                asdict(self._public_entity_placeholder_input(placeholder))
                for placeholder in payload.get("placeholders", [])
            ]
        elif kind == "patch_entity_states" and isinstance(payload.get("patches"), list):
            payload["patches"] = [
                asdict(self._public_entity_state_patch_input(patch))
                for patch in payload.get("patches", [])
            ]
        return WorldCommand(kind=kind, payload=payload)

    @classmethod
    def _coerce_json_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, list | tuple):
            return [cls._coerce_json_value(item) for item in value]
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("controller_state keys must be strings")
                normalized[key] = cls._coerce_json_value(item)
            return normalized
        raise TypeError(f"controller_state must be JSON-serializable, got {type(value).__name__}")

    @classmethod
    def _normalize_json_payload_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list | tuple):
            return [cls._normalize_json_payload_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_json_payload_value(item)
                for key, item in value.items()
            }
        return deepcopy(value)

    def _coerce_world_frame_input(self, frame_input: WorldFrameInput | dict[str, Any]) -> WorldFrameInput:
        if isinstance(frame_input, WorldFrameInput):
            controller_state_provided = bool(
                frame_input.controller_state_provided or frame_input.controller_state is not None
            )
            return WorldFrameInput(
                submission_id=None if frame_input.submission_id is None else int(frame_input.submission_id),
                focus_center=(
                    None
                    if frame_input.focus_center is None
                    else (int(frame_input.focus_center[0]), int(frame_input.focus_center[1]))
                ),
                controller_state=(
                    self._coerce_json_value(frame_input.controller_state)
                    if controller_state_provided
                    else None
                ),
                controller_state_provided=controller_state_provided,
                entities=None
                if frame_input.entities is None
                else [self._public_entity_state_input(entity) for entity in frame_input.entities],
                entity_placeholders=None
                if frame_input.entity_placeholders is None
                else [self._public_entity_placeholder_input(item) for item in frame_input.entity_placeholders],
                force_sources=None
                if frame_input.force_sources is None
                else [self._public_force_source_input(source) for source in frame_input.force_sources],
                emitters=None
                if frame_input.emitters is None
                else [self._coerce_emitter(emitter) for emitter in frame_input.emitters],
                target_queries=[self._coerce_target_query(query) for query in frame_input.target_queries],
                change_intents=[self._coerce_change_intent(intent) for intent in frame_input.change_intents],
                carrier_intents=[self._coerce_carrier_intent(intent) for intent in frame_input.carrier_intents],
                observation_targets=[
                    self._coerce_observation_target(target) for target in frame_input.observation_targets
                ],
                readback_requests=[
                    self._coerce_readback_request(request) for request in frame_input.readback_requests
                ],
                commands=[self._coerce_world_command(command) for command in frame_input.commands],
            )
        payload = dict(frame_input)
        focus_center = payload.get("focus_center")
        controller_state_provided = bool(payload.get("controller_state_provided", False)) or "controller_state" in payload
        return WorldFrameInput(
            submission_id=None if payload.get("submission_id") is None else int(payload["submission_id"]),
            focus_center=None if focus_center is None else (int(focus_center[0]), int(focus_center[1])),
            controller_state=(
                self._coerce_json_value(payload.get("controller_state"))
                if controller_state_provided
                else None
            ),
            controller_state_provided=controller_state_provided,
            entities=None
            if payload.get("entities") is None
            else [self._public_entity_state_input(entity) for entity in payload.get("entities", [])],
            entity_placeholders=None
            if payload.get("entity_placeholders") is None
            else [self._public_entity_placeholder_input(item) for item in payload.get("entity_placeholders", [])],
            force_sources=None
            if payload.get("force_sources") is None
            else [self._public_force_source_input(source) for source in payload.get("force_sources", [])],
            emitters=None
            if payload.get("emitters") is None
            else [self._coerce_emitter(emitter) for emitter in payload.get("emitters", [])],
            target_queries=[self._coerce_target_query(query) for query in payload.get("target_queries", [])],
            change_intents=[self._coerce_change_intent(intent) for intent in payload.get("change_intents", [])],
            carrier_intents=[self._coerce_carrier_intent(intent) for intent in payload.get("carrier_intents", [])],
            observation_targets=[self._coerce_observation_target(target) for target in payload.get("observation_targets", [])],
            readback_requests=[self._coerce_readback_request(request) for request in payload.get("readback_requests", [])],
            commands=[self._coerce_world_command(command) for command in payload.get("commands", [])],
        )

    def _gas_field_count(self) -> int:
        return max(self.rulebook.gases_by_id, default=-1) + 1

    def _light_field_count(self) -> int:
        max_light_id = max(self.rulebook.lights_by_id, default=-1)
        max_dose_channel = max((int(light.dose_channel_id) for light in self.rulebook.lights_by_id.values()), default=-1)
        return max(max_light_id, max_dose_channel) + 1

    def update_material_table(self, materials: list[MaterialDef | dict[str, Any]], *, immediate: bool = True) -> None:
        materials = [self._coerce_material_def(material) for material in materials]
        if not immediate:
            self.queue_command("update_material_table", materials=[asdict(material) for material in materials])
            return
        merged_payload = self._merged_material_table_payload(materials)
        self._validate_material_table_payload(merged_payload)
        self.rulebook.materials_by_name.clear()
        self.rulebook.materials_by_id.clear()
        self.rulebook.update_materials(self._coerce_material_def(item) for item in merged_payload)
        self.rulebook.optics.clear()
        self.rulebook.update_optics(
            build_default_optics_entries(
                self.rulebook.materials_by_id.values(),
                self.rulebook.lights_by_id.values(),
                existing=self._material_optics_snapshot_map(),
            )
        )
        self.tag_bits_by_name = deepcopy(self.rulebook.tag_bits)
        self._rebuild_material_property_arrays()
        optics_payload = self._material_optics_table_snapshot_payload()
        self._set_stable_shadow_payload("materials", merged_payload)
        self._set_stable_shadow_payload("optics", optics_payload)
        self.bridge.upload_table("materials", merged_payload)
        self.bridge.upload_table("optics", optics_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_material_table")
        self.bridge.ensure_world_resources(self)

    def update_gas_species_table(self, gases: list[GasSpeciesDef | dict[str, Any]], *, immediate: bool = True) -> None:
        gases = [self._coerce_gas_species_def(gas) for gas in gases]
        if not immediate:
            self.queue_command("update_gas_species_table", gases=[asdict(gas) for gas in gases])
            return
        merged_payload = self._merged_gas_species_table_payload(gases)
        self._validate_gas_species_payload(merged_payload)
        self.rulebook.gases_by_name.clear()
        self.rulebook.gases_by_id.clear()
        self.rulebook.update_gases(self._coerce_gas_species_def(item) for item in merged_payload)
        self._rebuild_gas_property_arrays()
        previous = self.gas_concentration
        gas_count = self._gas_field_count()
        self.gas_concentration = np.zeros((gas_count, self.gas_height, self.gas_width), dtype=np.float32)
        count = min(previous.shape[0], self.gas_concentration.shape[0])
        self.gas_concentration[:count] = previous[:count]
        if 0 <= self.air_gas_species_id < self.gas_concentration.shape[0]:
            self.gas_concentration[self.air_gas_species_id] = np.maximum(
                self.gas_concentration[self.air_gas_species_id], 1.0
            )
        self._set_stable_shadow_payload("gases", merged_payload)
        self.bridge.upload_table("gases", merged_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_gas_species_table")
        self.bridge.ensure_world_resources(self)

    def update_light_type_table(self, lights: list[LightTypeDef | dict[str, Any]], *, immediate: bool = True) -> None:
        lights = [self._coerce_light_type_def(light) for light in lights]
        if not immediate:
            self.queue_command("update_light_type_table", lights=[asdict(light) for light in lights])
            return
        merged_payload = self._merged_light_type_table_payload(lights)
        self._validate_light_type_payload(merged_payload)
        self.rulebook.lights_by_name.clear()
        self.rulebook.lights_by_id.clear()
        self.rulebook.update_lights(self._coerce_light_type_def(item) for item in merged_payload)
        self.rulebook.optics.clear()
        self.rulebook.update_optics(
            build_default_optics_entries(
                self.rulebook.materials_by_id.values(),
                self.rulebook.lights_by_id.values(),
                existing=self._material_optics_snapshot_map(),
            )
        )
        self._rebuild_light_property_arrays()
        previous_cell_dose = self.cell_optical_dose
        previous_gas_dose = self.gas_optical_dose
        light_count = self._light_field_count()
        self.cell_optical_dose = np.zeros((light_count, self.height, self.width), dtype=np.float32)
        self.gas_optical_dose = np.zeros((light_count, self.gas_height, self.gas_width), dtype=np.float32)
        cell_count = min(previous_cell_dose.shape[0], self.cell_optical_dose.shape[0])
        gas_count = min(previous_gas_dose.shape[0], self.gas_optical_dose.shape[0])
        self.cell_optical_dose[:cell_count] = previous_cell_dose[:cell_count]
        self.gas_optical_dose[:gas_count] = previous_gas_dose[:gas_count]
        optics_payload = self._material_optics_table_snapshot_payload()
        self._set_stable_shadow_payload("lights", merged_payload)
        self._set_stable_shadow_payload("optics", optics_payload)
        self.bridge.upload_table("lights", merged_payload)
        self.bridge.upload_table("optics", optics_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_light_type_table")
        self.bridge.ensure_world_resources(self)

    def update_material_optics_table(self, optics: list[MaterialOpticsDef | dict[str, Any]], *, immediate: bool = True) -> None:
        optics = [self._coerce_material_optics_def(entry) for entry in optics]
        if not immediate:
            self.queue_command("update_material_optics_table", optics=[asdict(entry) for entry in optics])
            return
        merged_payload = self._merged_material_optics_table_payload(optics)
        self._validate_material_optics_payload(merged_payload)
        self.rulebook.optics.clear()
        self.rulebook.update_optics(self._coerce_material_optics_def(item) for item in merged_payload)
        self._set_stable_shadow_payload("optics", merged_payload)
        self.bridge.upload_table("optics", merged_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_material_optics_table")

    def update_reaction_table(
        self,
        actions: list[ReactionAction | dict[str, Any]],
        rules: dict[str, object],
        *,
        immediate: bool = True,
    ) -> None:
        actions = [self._coerce_reaction_action(action) for action in actions]
        rules = self._coerce_reaction_rules(rules)
        if not immediate:
            self.queue_command(
                "update_reaction_table",
                actions=[asdict(action) for action in actions],
                rules={name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
            )
            return
        merged_payload = self._merged_reaction_table_payload(actions, rules)
        self._validate_reaction_payload(merged_payload)
        self.rulebook.reaction_actions = [ReactionAction(ReactionType.NONE)] + [
            self._coerce_reaction_action(action) for action in merged_payload["actions"]
        ]
        merged_rules = merged_payload["rules"]
        self.rulebook.material_material_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_material"]
        ]
        self.rulebook.material_gas_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_gas"]
        ]
        self.rulebook.material_light_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["material_light"]
        ]
        self.rulebook.gas_gas_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["gas_gas"]
        ]
        self.rulebook.gas_light_rules = [
            self._coerce_pair_reaction_rule(rule) for rule in merged_rules["gas_light"]
        ]
        self.rulebook.self_rules = [
            self._coerce_self_reaction_rule(rule) for rule in merged_rules["self_rules"]
        ]
        self._set_stable_shadow_payload("reactions", merged_payload)
        self.bridge.upload_table("reactions", merged_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("update_reaction_table")

    def replace_reaction_table(
        self,
        actions: list[ReactionAction | dict[str, Any]],
        rules: dict[str, object],
        *,
        immediate: bool = True,
    ) -> None:
        actions = [self._coerce_reaction_action(action) for action in actions]
        rules = self._coerce_reaction_rules(rules)
        if not immediate:
            self.queue_command(
                "replace_reaction_table",
                actions=[asdict(action) for action in actions],
                rules={name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
            )
            return
        replacement_payload = {
            "actions": [asdict(action) for action in actions],
            "rules": {name: [asdict(rule) for rule in entries] for name, entries in rules.items()},
        }
        self._validate_reaction_payload(replacement_payload)
        materials_payload = self._shadow_material_payload()
        self._clamp_material_payload_reaction_slots(
            materials_payload,
            action_count=len(replacement_payload["actions"]) + 1,
        )
        self.rulebook.reaction_actions = [ReactionAction(ReactionType.NONE)] + list(actions)
        self.rulebook.material_material_rules = list(rules["material_material"])
        self.rulebook.material_gas_rules = list(rules["material_gas"])
        self.rulebook.material_light_rules = list(rules["material_light"])
        self.rulebook.gas_gas_rules = list(rules["gas_gas"])
        self.rulebook.gas_light_rules = list(rules["gas_light"])
        self.rulebook.self_rules = list(rules["self_rules"])
        self.rulebook.materials_by_name.clear()
        self.rulebook.materials_by_id.clear()
        self.rulebook.update_materials(self._coerce_material_def(item) for item in materials_payload)
        self.tag_bits_by_name = deepcopy(self.rulebook.tag_bits)
        self._rebuild_material_property_arrays()
        reaction_payload = self._reaction_table_snapshot_payload()
        self._set_stable_shadow_payload("materials", materials_payload)
        self._set_stable_shadow_payload("reactions", reaction_payload)
        self.bridge.upload_table("materials", materials_payload)
        self.bridge.upload_table("reactions", reaction_payload)
        self.bridge.sync_rule_tables(self)
        self.bootstrap_log.append("replace_reaction_table")

    def reset_world(self, *, immediate: bool = True) -> None:
        if not immediate:
            self.queue_command("reset_world")
            return
        self._reset_world_state(reset_bridge_frame_inputs=True, keep_command_log=False)

    def _reset_world_state(
        self,
        *,
        reset_bridge_frame_inputs: bool,
        keep_command_log: bool,
    ) -> None:
        self.material_id.fill(0)
        self.phase.fill(0)
        self.cell_flags.fill(0)
        self.velocity.fill(0.0)
        self.cell_temperature.fill(20.0)
        self.timer_pack.fill(0)
        self.integrity.fill(0.0)
        self.island_id.fill(0)
        self.entity_id.fill(0)
        self.placeholder_displaced_material.fill(0)
        self.collapse_delay_pending.fill(False)
        self.flow_velocity.fill(0.0)
        self.ambient_temperature.fill(20.0)
        self.pressure_ping.fill(0.0)
        self.gas_concentration.fill(0.0)
        if 0 <= self.air_gas_species_id < self.gas_concentration.shape[0]:
            self.gas_concentration[self.air_gas_species_id] = 1.0
        self.visible_illumination.fill(0.0)
        self.cell_optical_dose.fill(0.0)
        self.gas_optical_dose.fill(0.0)
        self.force_sources.clear()
        self.persistent_emitters.clear()
        self.emitters.clear()
        self.collapse_dirty_regions.clear()
        self.collapse_deferred_regions.clear()
        clear_collapse_structure_dirty_tile_mask(self)
        self.islands.clear()
        self.entity_states.clear()
        self.entity_placeholders.clear()
        self.pending_frame_inputs.clear()
        self.completed_frame_outputs.clear()
        self.canceled_frame_submission_ids.clear()
        self.next_frame_submission_id = 1
        self.next_readback_request_id = 1
        self.pending_readbacks.clear()
        self.inflight_readbacks.clear()
        self.completed_readbacks.clear()
        self.canceled_readback_request_ids.clear()
        self.last_entity_observation_consume_snapshot = {
            "frame_id": int(self.frame_id),
            "consumed": 0,
            "consumed_readbacks": [],
            "observations": {},
            "entity_feedback": {},
        }
        self.controller_state_snapshot = None
        self.gas_solver.reset_runtime_state(self)
        self.heat_solver.reset_runtime_state(self)
        self.liquid_solver.reset_runtime_state(self)
        self.reaction_solver.reset_runtime_state(self)
        self.collapse_solver.reset_runtime_state(self)
        self.optics_solver.reset_runtime_state(self)
        self.motion_solver.reset_runtime_state()
        if reset_bridge_frame_inputs:
            self._clear_bridge_frame_inputs(keep_commands=keep_command_log, prepared=False)
        self.page_store.clear()
        self.next_island_id = 1
        self._build_demo_scene()

    def queue_command(self, kind: str, **payload: Any) -> None:
        self.command_queue.append(WorldCommand(kind=kind, payload=deepcopy(payload)))

    def _resolve_direct_targeted_coords(
        self,
        kind: str,
        x: int | None,
        y: int | None,
        *,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> tuple[int, int, str | None]:
        fields = TARGETED_COMMAND_COORD_FIELDS.get(kind)
        if fields is None:
            raise ValueError(f"unsupported direct target query kind '{kind}'")
        x_field, y_field = fields
        if target_queries is None:
            if x is None or y is None:
                raise ValueError(f"{x_field} and {y_field} are required unless target_queries resolve target_query_id")
            return int(x), int(y), None
        if target_query_id is None:
            raise ValueError("target_query_id is required when target_queries are provided")
        resolved_targets = self._resolve_target_queries(
            [self._coerce_target_query(query) for query in target_queries]
        )
        resolved_commands = self._resolve_targeted_commands(
            [
                WorldCommand(
                    kind=kind,
                    payload={
                        "target_query_id": str(target_query_id),
                        "target_dx": int(target_dx),
                        "target_dy": int(target_dy),
                    },
                )
            ],
            resolved_targets,
        )
        if not resolved_commands:
            raise ValueError(f"unable to resolve {kind} target query")
        payload = resolved_commands[0].payload
        return int(payload[x_field]), int(payload[y_field]), str(payload.get("resolved_target_query_id", target_query_id))

    def inject_material(
        self,
        x: int | None,
        y: int | None,
        material: str,
        radius: int = 2,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_material",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [WorldCommand(kind="inject_material", payload={"x": int(x), "y": int(y), "material": material, "radius": radius})]
            )
        else:
            payload: dict[str, Any] = {"x": x, "y": y, "material": material, "radius": radius}
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_material", **payload)

    def write_material_region(
        self,
        x: int | None,
        y: int | None,
        width: int,
        height: int,
        material: str,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "write_material_region",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [
                    WorldCommand(
                        kind="write_material_region",
                        payload={"x": int(x), "y": int(y), "width": width, "height": height, "material": material},
                    )
                ]
            )
        else:
            payload: dict[str, Any] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "material": material,
            }
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("write_material_region", **payload)

    def inject_temperature(
        self,
        x: int | None,
        y: int | None,
        delta: float,
        radius: int = 2,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_temperature",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [WorldCommand(kind="inject_temperature", payload={"x": int(x), "y": int(y), "delta": delta, "radius": radius})]
            )
        else:
            payload: dict[str, Any] = {"x": x, "y": y, "delta": delta, "radius": radius}
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_temperature", **payload)

    def inject_velocity(
        self,
        x: int | None,
        y: int | None,
        velocity: tuple[float, float],
        radius: int = 2,
        *,
        carrier: str = "cell",
        mode: str = "add",
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_velocity",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [
                    WorldCommand(
                        kind="inject_velocity",
                        payload={
                            "x": int(x),
                            "y": int(y),
                            "velocity": velocity,
                            "radius": radius,
                            "carrier": carrier,
                            "mode": mode,
                        },
                    )
                ]
            )
        else:
            payload: dict[str, Any] = {
                "x": x,
                "y": y,
                "velocity": velocity,
                "radius": radius,
                "carrier": carrier,
                "mode": mode,
            }
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_velocity", **payload)

    def inject_force(
        self,
        x: int | None,
        y: int | None,
        direction: tuple[float, float],
        radius: float,
        strength: float,
        lifetime: float = 0.5,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_force",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            world_x = float(x)
            world_y = float(y)
            x, y = self._world_to_buffer_clamped(int(x), int(y))
            self._append_force_source_immediate(
                ForceSource(
                    x=float(x),
                    y=float(y),
                    direction=(float(direction[0]), float(direction[1])),
                    radius=float(radius),
                    strength=float(strength),
                    lifetime=float(lifetime),
                    world_x=world_x,
                    world_y=world_y,
                )
            )
            return
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "direction": direction,
            "radius": radius,
            "strength": strength,
            "lifetime": lifetime,
        }
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        self.queue_command("inject_force", **payload)

    def inject_gas(
        self,
        x: int | None,
        y: int | None,
        species: str,
        amount: float,
        radius: int = 1,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_gas",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            self._apply_grid_world_commands(
                [
                    WorldCommand(
                        kind="inject_gas",
                        payload={"x": int(x), "y": int(y), "species": species, "amount": amount, "radius": radius},
                    )
                ]
            )
        else:
            payload: dict[str, Any] = {"x": x, "y": y, "species": species, "amount": amount, "radius": radius}
            if resolved_target_query_id is not None:
                payload["resolved_target_query_id"] = resolved_target_query_id
            self.queue_command("inject_gas", **payload)

    def request_readback(
        self,
        center_x: int | None,
        center_y: int | None,
        width: int,
        height: int,
        channels: tuple[str, ...],
        *,
        request_id: int | None = None,
        observer_id: int | None = None,
        label: str | None = None,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> int:
        request = ReadbackRequest(
            request_id=request_id,
            center_x=None if center_x is None else int(center_x),
            center_y=None if center_y is None else int(center_y),
            width=int(width),
            height=int(height),
            channels=tuple(channels),
            observer_id=observer_id,
            label=label,
            target_query_id=None if target_query_id is None else str(target_query_id),
            target_dx=int(target_dx),
            target_dy=int(target_dy),
        )
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
            resolved_request = self._resolve_readback_request(request, resolved_targets)
            if resolved_request is None:
                raise ValueError("unable to resolve readback request target query")
            request = resolved_request
        request = self._normalize_readback_request(request)
        if request.center_x is None or request.center_y is None:
            raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
        request = self._assign_readback_request_id(request)
        self.queue_command(
            "request_readback",
            request_id=request.request_id,
            center_x=request.center_x,
            center_y=request.center_y,
            width=request.width,
            height=request.height,
            channels=request.channels,
            observer_id=request.observer_id,
            label=request.label,
            target_query_id=request.target_query_id,
            target_dx=int(request.target_dx),
            target_dy=int(request.target_dy),
        )
        assert request.request_id is not None
        return int(request.request_id)

    def preview_readback(
        self,
        center_x: int | None,
        center_y: int | None,
        width: int,
        height: int,
        channels: tuple[str, ...],
        *,
        request_id: int | None = None,
        observer_id: int | None = None,
        label: str | None = None,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ReadbackRequest:
        request = ReadbackRequest(
            request_id=request_id,
            center_x=None if center_x is None else int(center_x),
            center_y=None if center_y is None else int(center_y),
            width=int(width),
            height=int(height),
            channels=tuple(channels),
            observer_id=observer_id,
            label=label,
            target_query_id=None if target_query_id is None else str(target_query_id),
            target_dx=int(target_dx),
            target_dy=int(target_dy),
        )
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
            resolved_request = self._resolve_readback_request(request, resolved_targets)
            if resolved_request is None:
                raise ValueError("unable to resolve readback request target query")
            request = resolved_request
        request = self._normalize_readback_request(request)
        if request.center_x is None or request.center_y is None:
            raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
        return request

    def request_observation(
        self,
        target: ObservationTarget | dict[str, Any],
        *,
        request_id: int | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> int:
        target = self._coerce_observation_target(target)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        request = self._build_observation_request(target, resolved_targets)
        if request is None:
            if target.target_query_id is not None and target_queries is not None:
                raise ValueError("unable to resolve observation target query")
            raise ValueError("unable to resolve observation target")
        if request_id is not None:
            request = replace(request, request_id=int(request_id))
        request = self._assign_readback_request_id(request)
        self.queue_command(
            "request_readback",
            request_id=request.request_id,
            center_x=request.center_x,
            center_y=request.center_y,
            width=request.width,
            height=request.height,
            channels=request.channels,
            observer_id=request.observer_id,
            label=request.label,
            target_query_id=request.target_query_id,
            target_dx=int(request.target_dx),
            target_dy=int(request.target_dy),
        )
        assert request.request_id is not None
        return int(request.request_id)

    def preview_observation(
        self,
        target: ObservationTarget | dict[str, Any],
        *,
        request_id: int | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ReadbackRequest:
        target = self._coerce_observation_target(target)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        request = self._build_observation_request(target, resolved_targets)
        if request is None:
            if target.target_query_id is not None and target_queries is not None:
                raise ValueError("unable to resolve observation target query")
            raise ValueError("unable to resolve observation target")
        if request_id is not None:
            request = replace(request, request_id=int(request_id))
        return request

    def _resolve_public_world_command(
        self,
        command: WorldCommand | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None,
        assign_readback_request_id: bool,
    ) -> WorldCommand:
        command = self._coerce_world_command(command)
        if command.kind not in PUBLIC_WORLD_COMMAND_KINDS:
            raise ValueError(f"unsupported public world command kind '{command.kind}'")

        if command.kind == "request_readback":
            request = self._coerce_readback_request(command.payload)
            if target_queries is not None:
                resolved_targets = self._resolve_target_queries(
                    [self._coerce_target_query(query) for query in target_queries]
                )
                resolved_request = self._resolve_readback_request(request, resolved_targets)
                if resolved_request is None:
                    raise ValueError("unable to resolve world command target query")
                request = resolved_request
            elif request.target_query_id is not None and (request.center_x is None or request.center_y is None):
                raise ValueError("target_queries are required to resolve world command target_query_id")
            request = self._normalize_readback_request(request)
            if request.center_x is None or request.center_y is None:
                raise ValueError("center_x and center_y are required unless target_queries resolve target_query_id")
            if assign_readback_request_id:
                request = self._assign_readback_request_id(request)
            return WorldCommand(
                kind="request_readback",
                payload={
                    "request_id": request.request_id,
                    "center_x": request.center_x,
                    "center_y": request.center_y,
                    "width": request.width,
                    "height": request.height,
                    "channels": request.channels,
                    "observer_id": request.observer_id,
                    "label": request.label,
                    "target_query_id": request.target_query_id,
                    "target_dx": int(request.target_dx),
                    "target_dy": int(request.target_dy),
                },
            )

        if target_queries is None:
            if command.payload.get("target_query_id") is not None:
                raise ValueError("target_queries are required to resolve world command target_query_id")
            return command
        resolved_targets = self._resolve_target_queries(
            [self._coerce_target_query(query) for query in target_queries]
        )
        resolved_commands = self._resolve_targeted_commands([command], resolved_targets)
        if not resolved_commands:
            raise ValueError("unable to resolve world command target query")
        return resolved_commands[0]

    def _public_world_command(self, command: WorldCommand) -> WorldCommand:
        payload = deepcopy(command.payload)
        if command.kind == "sync_entity_states" and isinstance(payload, dict):
            entities = payload.get("entities")
            if isinstance(entities, list):
                payload["entities"] = [
                    self.serialize_entity_state_input(
                        entity if isinstance(entity, EntityState) else self._coerce_entity_state(entity)
                    )
                    for entity in entities
                ]
        elif command.kind == "patch_entity_states" and isinstance(payload, dict):
            patches = payload.get("patches")
            if isinstance(patches, list):
                payload["patches"] = [
                    self.serialize_entity_state_patch(
                        patch if isinstance(patch, EntityStatePatch) else self._coerce_entity_state_patch(patch)
                    )
                    for patch in patches
                ]
        elif command.kind == "sync_entity_placeholders" and isinstance(payload, dict):
            placeholders = payload.get("placeholders")
            if isinstance(placeholders, list):
                payload["placeholders"] = [
                    self.serialize_entity_placeholder_input(
                        placeholder
                        if isinstance(placeholder, EntityPlaceholder)
                        else self._coerce_entity_placeholder(placeholder)
                    )
                    for placeholder in placeholders
                ]
        return WorldCommand(kind=command.kind, payload=payload)

    def preview_world_command(
        self,
        command: WorldCommand | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> WorldCommand:
        command = self._coerce_world_command(command)
        resolved_command = self._resolve_public_world_command(
            command,
            target_queries=target_queries,
            assign_readback_request_id=False,
        )
        return self._public_world_command(resolved_command)

    def preview_target_queries(
        self,
        target_queries: list[TargetQuery | dict[str, Any]],
    ) -> dict[str, ResolvedTarget]:
        return {
            query_id: self._public_resolved_target(target)
            for query_id, target in self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            ).items()
        }

    def request_world_command(
        self,
        command: WorldCommand | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> WorldCommand:
        resolved_command = self._resolve_public_world_command(
            command,
            target_queries=target_queries,
            assign_readback_request_id=True,
        )
        self.queue_command(resolved_command.kind, **resolved_command.payload)
        return self._public_world_command(resolved_command)

    def preview_change_intent(
        self,
        intent: ChangeIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedChangeIntent:
        intent = self._coerce_change_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        return self._public_resolved_change_intent(self._resolve_change_intent(intent, resolved_targets))

    def request_change_intent(
        self,
        intent: ChangeIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedChangeIntent:
        intent = self._coerce_change_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        resolved_intent = self._resolve_change_intent(intent, resolved_targets)
        for command in resolved_intent.generated_commands:
            self.queue_command(command.kind, **command.payload)
        return self._public_resolved_change_intent(resolved_intent)

    def preview_carrier_intent(
        self,
        intent: CarrierIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedCarrierIntent:
        intent = self._coerce_carrier_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        return self._public_resolved_carrier_intent(self._resolve_carrier_intent(intent, resolved_targets))

    def request_carrier_intent(
        self,
        intent: CarrierIntent | dict[str, Any],
        *,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> ResolvedCarrierIntent:
        intent = self._coerce_carrier_intent(intent)
        resolved_targets: dict[str, ResolvedTarget] = {}
        if target_queries is not None:
            resolved_targets = self._resolve_target_queries(
                [self._coerce_target_query(query) for query in target_queries]
            )
        resolved_intent = self._resolve_carrier_intent(intent, resolved_targets)
        for command in resolved_intent.generated_commands:
            self.queue_command(command.kind, **command.payload)
        return self._public_resolved_carrier_intent(resolved_intent)

    def preview_frame_input(
        self,
        frame_input: WorldFrameInput | dict[str, Any],
        *,
        reserved_readback_request_ids: set[int] | None = None,
    ) -> WorldFramePreview:
        frame_input = self._coerce_world_frame_input(frame_input)
        saved_paging = deepcopy(self.paging)
        saved_preview_runtime = self._snapshot_preview_runtime_state()
        saved_entity_states = dict(self.entity_states)
        saved_entity_placeholders = {entity_id: set(cells) for entity_id, cells in self.entity_placeholders.items()}
        saved_controller_state = deepcopy(self.controller_state_snapshot)
        saved_blocked_cells = None if self._resolver_blocked_cells is None else set(self._resolver_blocked_cells)
        saved_released_cells = None if self._resolver_released_cells is None else set(self._resolver_released_cells)
        try:
            preview_controller_state = deepcopy(self.controller_state_snapshot)
            if frame_input.controller_state_provided:
                self.controller_state_snapshot = deepcopy(frame_input.controller_state)
                preview_controller_state = deepcopy(self.controller_state_snapshot)
            (
                paging_updates,
                preview_page_stripes,
                entity_observation_targets,
                placeholder_inputs,
                placeholder_count,
            ) = self._prepare_preview_frame_context(frame_input)
            resolved_targets = self._resolve_target_queries(frame_input.target_queries)
            resolved_change_intents, generated_commands = self._resolve_change_intents(frame_input.change_intents, resolved_targets)
            resolved_carrier_intents, generated_carrier_commands = self._resolve_carrier_intents(
                frame_input.carrier_intents,
                resolved_targets,
            )
            observation_pairs = self._build_observation_request_pairs(
                entity_observation_targets + frame_input.observation_targets,
                resolved_targets,
            )
            observation_requests, next_preview_request_id = self._assign_preview_readback_request_ids(
                [request for _, request in observation_pairs]
            )
            observation_pairs = [
                (target, request)
                for (target, _), request in zip(observation_pairs, observation_requests, strict=False)
            ]
            resolved_commands = (
                generated_commands
                + generated_carrier_commands
                + self._resolve_targeted_commands(frame_input.commands, resolved_targets)
            )
            readback_requests, _ = self._assign_preview_readback_request_ids(
                self._resolve_readback_requests(frame_input.readback_requests, resolved_targets),
                next_request_id=next_preview_request_id,
            )
            bridge_frame_snapshot = self._serialize_preview_bridge_frame_snapshot(
                current_entity_placeholders=saved_entity_placeholders,
                resolved_commands=resolved_commands,
                observation_requests=observation_requests,
                readback_requests=readback_requests,
                placeholder_inputs=placeholder_inputs,
                paging_updates=paging_updates,
                page_stripes=preview_page_stripes,
                reserved_readback_request_ids=reserved_readback_request_ids,
            )
            return WorldFramePreview(
                controller_state=preview_controller_state,
                resolved_targets={
                    query_id: self._public_resolved_target(target)
                    for query_id, target in resolved_targets.items()
                },
                resolved_change_intents={
                    intent_id: self._public_resolved_change_intent(intent)
                    for intent_id, intent in resolved_change_intents.items()
                },
                resolved_carrier_intents={
                    intent_id: self._public_resolved_carrier_intent(intent)
                    for intent_id, intent in resolved_carrier_intents.items()
                },
                resolved_commands=[self._public_world_command(command) for command in resolved_commands],
                observation_requests=observation_requests,
                observation_plans=[
                    self._serialize_observation_plan_for_target_request(target, request)
                    for target, request in observation_pairs
                ],
                readback_requests=readback_requests,
                readback_plans=self._serialize_readback_plans_for_requests(readback_requests),
                bridge_frame_snapshot=bridge_frame_snapshot,
                paging_updates=paging_updates,
                placeholder_count=placeholder_count,
            )
        finally:
            self._restore_preview_runtime_state(saved_preview_runtime)
            self.paging = saved_paging
            self.entity_states = saved_entity_states
            self.entity_placeholders = saved_entity_placeholders
            self.controller_state_snapshot = saved_controller_state
            self._resolver_blocked_cells = saved_blocked_cells
            self._resolver_released_cells = saved_released_cells

    def submit_frame_input(self, frame_input: WorldFrameInput | dict[str, Any]) -> int:
        frame_input = self._coerce_world_frame_input(frame_input)
        submission_id = frame_input.submission_id
        if submission_id is None:
            submission_id = self.next_frame_submission_id
        frame_input = replace(
            frame_input,
            submission_id=submission_id,
            readback_requests=[self._assign_readback_request_id(request) for request in frame_input.readback_requests],
        )
        self.next_frame_submission_id = max(self.next_frame_submission_id, int(submission_id) + 1)
        self.canceled_frame_submission_ids.discard(int(submission_id))
        self.pending_frame_inputs.append(frame_input)
        return int(submission_id)

    def request_frame_input(self, frame_input: WorldFrameInput | dict[str, Any]) -> dict[str, Any]:
        submission_id = self.submit_frame_input(frame_input)
        pending_frame_input = self._pending_frame_input(submission_id)
        preview = self.preview_frame_input(
            pending_frame_input,
            reserved_readback_request_ids=set(self._frame_readback_request_ids(pending_frame_input)),
        )
        return {
            "queued": True,
            "pending_frames": len(self.pending_frame_inputs),
            "submission_id": submission_id,
            "preview": preview,
        }

    def request_frame_cycle(
        self,
        frame_input: WorldFrameInput | dict[str, Any] | None = None,
        *,
        apply_frame: bool = True,
    ) -> dict[str, Any]:
        normalized_frame_input = {} if frame_input is None else frame_input
        preview = self.preview_frame_input(normalized_frame_input)
        if not apply_frame:
            return {
                "applied": False,
                "queued": False,
                "pending_frames": len(self.pending_frame_inputs),
                "submission_id": None,
                "preview": preview,
                "result": None,
            }
        submission_id = self.submit_frame_input(normalized_frame_input)
        pending_frame_input = self._pending_frame_input(submission_id)
        preview = self.preview_frame_input(
            pending_frame_input,
            reserved_readback_request_ids=set(self._frame_readback_request_ids(pending_frame_input)),
        )
        return {
            "applied": True,
            "queued": True,
            "pending_frames": len(self.pending_frame_inputs),
            "submission_id": submission_id,
            "preview": preview,
            "result": None,
        }

    def pending_frame_submission_ids(self) -> list[int]:
        return [int(frame_input.submission_id) for frame_input in self.pending_frame_inputs if frame_input.submission_id is not None]

    def _pending_frame_input(self, submission_id: int) -> WorldFrameInput:
        for frame_input in reversed(self.pending_frame_inputs):
            if frame_input.submission_id == int(submission_id):
                return frame_input
        raise KeyError(f"missing pending frame submission_id={submission_id}")

    @staticmethod
    def _frame_readback_request_ids(frame_input: WorldFrameInput) -> list[int]:
        return [
            int(request.request_id)
            for request in frame_input.readback_requests
            if request.request_id is not None
        ]

    def cancel_frame_submission(self, submission_id: int) -> bool:
        for index, frame_input in enumerate(self.pending_frame_inputs):
            if frame_input.submission_id == submission_id:
                del self.pending_frame_inputs[index]
                self.canceled_frame_submission_ids.add(int(submission_id))
                self.canceled_readback_request_ids.update(self._frame_readback_request_ids(frame_input))
                return True
        return False

    def cancel_all_pending_frame_submissions(self) -> list[int]:
        canceled = self.pending_frame_submission_ids()
        canceled_readback_ids: list[int] = []
        for frame_input in self.pending_frame_inputs:
            canceled_readback_ids.extend(self._frame_readback_request_ids(frame_input))
        self.pending_frame_inputs.clear()
        self.canceled_frame_submission_ids.update(canceled)
        self.canceled_readback_request_ids.update(canceled_readback_ids)
        return canceled

    def cancel_readback_request(self, request_id: int) -> bool:
        request_id = int(request_id)
        canceled = False

        remaining_commands: deque[WorldCommand] = deque()
        for command in self.command_queue:
            if command.kind == "request_readback" and int(command.payload.get("request_id", -1)) == request_id:
                canceled = True
                continue
            remaining_commands.append(command)
        self.command_queue = remaining_commands

        remaining_frames: deque[WorldFrameInput] = deque()
        for frame_input in self.pending_frame_inputs:
            remaining_readbacks = [request for request in frame_input.readback_requests if request.request_id != request_id]
            if len(remaining_readbacks) != len(frame_input.readback_requests):
                frame_input = replace(frame_input, readback_requests=remaining_readbacks)
                canceled = True
            remaining_frames.append(frame_input)
        self.pending_frame_inputs = remaining_frames

        next_pending = [request for request in self.pending_readbacks if request.request_id != request_id]
        if len(next_pending) != len(self.pending_readbacks):
            canceled = True
        self.pending_readbacks = next_pending

        next_inflight = [request for request in self.inflight_readbacks if request.request_id != request_id]
        if len(next_inflight) != len(self.inflight_readbacks):
            canceled = True
        self.inflight_readbacks = next_inflight

        next_completed = deque(
            result for result in self.completed_readbacks if result.request.request_id != request_id
        )
        if len(next_completed) != len(self.completed_readbacks):
            canceled = True
        self.completed_readbacks = next_completed

        if canceled:
            self.canceled_readback_request_ids.add(request_id)
        return canceled

    def poll_frame_output(self, submission_id: int | None = None) -> WorldFrameOutput | None:
        if submission_id is None:
            if not self.completed_frame_outputs:
                return None
            return self.completed_frame_outputs.popleft()
        for index, output in enumerate(self.completed_frame_outputs):
            if output.submission_id == submission_id:
                del self.completed_frame_outputs[index]
                return output
        return None

    def poll_all_frame_outputs(self) -> list[WorldFrameOutput]:
        outputs: list[WorldFrameOutput] = []
        while self.completed_frame_outputs:
            outputs.append(self.completed_frame_outputs.popleft())
        return outputs

    def serialize_pending_commands(self) -> dict[str, Any]:
        return {
            "pending": len(self.command_queue),
            "commands": [self.serialize_world_command(command) for command in self.command_queue],
        }

    def serialize_readback_state(self) -> dict[str, Any]:
        queued_commands = [
            self.serialize_world_command(command)
            for command in self.command_queue
            if command.kind == "request_readback"
        ]
        bridge_runtime = self.bridge.serialize_runtime_state()
        readback_slots = [
            slot
            for slot in bridge_runtime.get("readback_slots", [])
            if bool(slot.get("occupied", False))
        ]
        return {
            "queued": len(queued_commands),
            "queued_commands": queued_commands,
            "pending": len(self.pending_readbacks),
            "pending_requests": [self.serialize_readback_request(request) for request in self.pending_readbacks],
            "inflight": len(self.inflight_readbacks),
            "inflight_requests": [self.serialize_readback_request(request) for request in self.inflight_readbacks],
            "inflight_slots": readback_slots,
            "readback_latency_frames": bridge_runtime.get("readback_latency_frames", {}),
            "ready": len(self.completed_readbacks),
        }

    def serialize_bridge_runtime(self) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "bridge": self.bridge.serialize_runtime_state(),
            "pending_readbacks": len(self.pending_readbacks),
            "inflight_readbacks": len(self.inflight_readbacks),
            "ready_readbacks": len(self.completed_readbacks),
            "pending_commands": len(self.command_queue),
        }

    @staticmethod
    def _serialize_bridge_resource_summary(name: str, array: np.ndarray) -> dict[str, Any]:
        return {
            "name": str(name),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "structured": array.dtype.names is not None,
            "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
            "row_count": int(array.shape[0]) if array.ndim > 0 else 0,
        }

    def serialize_bridge_resources(self) -> dict[str, Any]:
        typed_tables = [
            {
                **self._serialize_bridge_resource_summary(str(name), table),
                "endpoint": "/api/read/bridge_typed_table",
                "query": {"name": str(name)},
                "response_type": "bridge_typed_table_snapshot",
                "slice_endpoint": "/api/read/bridge_typed_table_slice",
                "slice_query": {"name": str(name), "offset": 0, "limit": 64},
                "slice_response_type": "bridge_typed_table_slice_snapshot",
            }
            for name, table in sorted(self.bridge.shadow_typed_tables.items())
        ]
        shadow_buffers = []
        for name, buffer in sorted(self.bridge.shadow_buffers.items()):
            resource = {
                **self._serialize_bridge_resource_summary(str(name), buffer),
                "endpoint": "/api/read/bridge_shadow_buffer",
                "query": {"name": str(name)},
                "response_type": "bridge_shadow_buffer_snapshot",
                "slice_endpoint": "/api/read/bridge_shadow_buffer_slice",
                "slice_query": {"name": str(name), "offset": 0, "limit": 64},
                "slice_response_type": "bridge_shadow_buffer_slice_snapshot",
            }
            if buffer.ndim >= 2:
                resource["window_endpoint"] = "/api/read/bridge_shadow_buffer_window"
                resource["window_query"] = {"name": str(name), "x": 0, "y": 0, "w": 16, "h": 16}
                resource["window_response_type"] = "bridge_shadow_buffer_window_snapshot"
                resource["window_axes"] = [int(buffer.ndim - 2), int(buffer.ndim - 1)]
            trailing_shape = tuple(int(value) for value in buffer.shape[-2:]) if buffer.ndim >= 2 else ()
            if trailing_shape == (int(self.height), int(self.width)):
                resource["world_window_endpoint"] = "/api/read/bridge_shadow_buffer_world_window"
                resource["world_window_query"] = {"name": str(name), "x": int(self.paging.origin_x), "y": int(self.paging.origin_y), "w": 16, "h": 16}
                resource["world_window_response_type"] = "bridge_shadow_buffer_world_window_snapshot"
            if trailing_shape == (int(self.gas_height), int(self.gas_width)):
                resource["gas_window_endpoint"] = "/api/read/bridge_shadow_buffer_gas_window"
                resource["gas_window_query"] = {
                    "name": str(name),
                    "x": int(self.paging.origin_x) // int(self.gas_cell_size),
                    "y": int(self.paging.origin_y) // int(self.gas_cell_size),
                    "w": 4,
                    "h": 4,
                }
                resource["gas_window_response_type"] = "bridge_shadow_buffer_gas_window_snapshot"
            shadow_buffers.append(resource)
        return {
            "typed_tables": typed_tables,
            "shadow_buffers": shadow_buffers,
            "snapshots": [
                {
                    "name": "bridge_runtime",
                    "endpoint": "/api/read/bridge_runtime",
                    "response_type": "bridge_runtime",
                },
                {
                    "name": "bridge_uploads",
                    "endpoint": "/api/read/bridge_uploads",
                    "response_type": "bridge_upload_snapshot",
                },
                {
                    "name": "bridge_frame",
                    "endpoint": "/api/read/bridge_frame",
                    "response_type": "bridge_frame_snapshot",
                },
            ],
        }

    def serialize_ready_readbacks(self) -> dict[str, Any]:
        return {
            "ready": len(self.completed_readbacks),
            "results": [self.serialize_readback_result(result) for result in self.completed_readbacks],
        }

    def readback_request_status(self, request_id: int) -> str:
        if any(
            command.kind == "request_readback" and int(command.payload.get("request_id", -1)) == int(request_id)
            for command in self.command_queue
        ):
            return "queued"
        if any(
            any(request.request_id == int(request_id) for request in frame_input.readback_requests)
            for frame_input in self.pending_frame_inputs
        ):
            return "pending_frame"
        if any(request.request_id == int(request_id) for request in self.pending_readbacks):
            return "pending"
        if any(request.request_id == int(request_id) for request in self.inflight_readbacks):
            return "inflight"
        if any(result.request.request_id == int(request_id) for result in self.completed_readbacks):
            return "ready"
        if int(request_id) in self.canceled_readback_request_ids:
            return "canceled"
        return "missing"

    def serialize_frame_state(self) -> dict[str, Any]:
        pending_submission_ids = self.pending_frame_submission_ids()
        ready_submission_ids = [
            int(output.submission_id)
            for output in self.completed_frame_outputs
            if output.submission_id is not None
        ]
        return {
            "pending": len(self.pending_frame_inputs),
            "pending_submission_ids": pending_submission_ids,
            "ready": len(self.completed_frame_outputs),
            "ready_submission_ids": ready_submission_ids,
            "canceled_submission_ids": sorted(int(submission_id) for submission_id in self.canceled_frame_submission_ids),
        }

    def serialize_pending_frame_inputs(self) -> dict[str, Any]:
        return {
            "pending": len(self.pending_frame_inputs),
            "frames": [self.serialize_pending_frame_detail(frame_input) for frame_input in self.pending_frame_inputs],
        }

    def serialize_pending_frame_detail(self, frame_input: WorldFrameInput) -> dict[str, Any]:
        payload = self.serialize_frame_input(frame_input)
        payload["preview"] = self.serialize_frame_preview(
            self.preview_frame_input(
                frame_input,
                reserved_readback_request_ids=set(self._frame_readback_request_ids(frame_input)),
            )
        )
        return payload

    def serialize_ready_frame_outputs(self) -> dict[str, Any]:
        return {
            "ready": len(self.completed_frame_outputs),
            "outputs": [self.serialize_frame_output(output) for output in self.completed_frame_outputs],
        }

    def frame_submission_status(self, submission_id: int) -> str:
        if any(frame_input.submission_id == submission_id for frame_input in self.pending_frame_inputs):
            return "pending"
        if any(output.submission_id == submission_id for output in self.completed_frame_outputs):
            return "ready"
        if submission_id in self.canceled_frame_submission_ids:
            return "canceled"
        return "missing"

    def inject_light(
        self,
        x: int | None,
        y: int | None,
        light_type: str,
        strength: float,
        radius: int | None = None,
        *,
        direction: tuple[float, float] = (0.0, 0.0),
        spread: float = 0.25,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> None:
        x, y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "inject_light",
            x,
            y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        light_id = self._resolve_sanctioned_light_id(light_type)
        if light_id < 0:
            raise KeyError(light_type)
        if radius is not None:
            resolved_radius = int(radius)
        else:
            shadow_default_range = self._shadow_light_default_range(light_id)
            if shadow_default_range is None:
                raise KeyError(light_type)
            resolved_radius = int(shadow_default_range)
        if immediate:
            world_origin = (int(x), int(y))
            x, y = self._world_to_buffer_clamped(int(x), int(y))
            shadow_light = self._shadow_light_name(light_id)
            if shadow_light is None:
                raise KeyError(light_type)
            self._append_transient_light_emitter_immediate(
                {
                    "light_type": shadow_light,
                    "origin": (int(x), int(y)),
                    "world_origin": world_origin,
                    "direction": (float(direction[0]), float(direction[1])),
                    "spread": float(spread),
                    "strength": float(strength),
                    "range_cells": int(resolved_radius),
                }
            )
            return
        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "light_type": self._shadow_light_name(light_id),
            "strength": strength,
            "radius": resolved_radius,
            "direction": direction,
            "spread": spread,
        }
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        self.queue_command("inject_light", **payload)

    def focus_paging(self, center_x: int, center_y: int) -> list[PageStripeUpdate]:
        return self.paging.focus_on(center_x, center_y)

    def advance_paging(
        self,
        center_x: int | None,
        center_y: int | None,
        *,
        immediate: bool = False,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> list[PageStripeUpdate]:
        center_x, center_y, resolved_target_query_id = self._resolve_direct_targeted_coords(
            "advance_paging",
            center_x,
            center_y,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            return self._advance_paging(center_x, center_y)
        payload: dict[str, Any] = {"center_x": center_x, "center_y": center_y}
        if resolved_target_query_id is not None:
            payload["resolved_target_query_id"] = resolved_target_query_id
        self.queue_command("advance_paging", **payload)
        return []

    def capture_page_stripe(self, update: PageStripeUpdate) -> dict[str, Any]:
        update = self._contextualize_page_stripe_update(update)
        if self.simulation_backend == "gpu":
            if self._gpu_cpu_dirty_resources:
                self.bridge.sync_world(self)
                self._gpu_cpu_dirty_resources.clear()
            return self.page_stripe_pipeline.capture(self, update)
        return self._capture_page_stripe_cpu_snapshot(update)

    def _capture_page_stripe_cpu_snapshot(self, update: PageStripeUpdate) -> dict[str, Any]:
        gas_ranges = self._stripe_buffer_ranges(update, gas_grid=True)
        cell_axis = 1 if update.axis == "x" else 0
        cell_dose_axis = 2 if update.axis == "x" else 1
        material_id = self._capture_stripe_array(self.material_id, update, stripe_axis=cell_axis)
        phase = self._capture_stripe_array(self.phase, update, stripe_axis=cell_axis)
        island_id = self._capture_stripe_array(self.island_id, update, stripe_axis=cell_axis)
        entity_id = self._capture_stripe_array(self.entity_id, update, stripe_axis=cell_axis)
        placeholder_displaced_material = self._capture_stripe_array(
            self.placeholder_displaced_material,
            update,
            stripe_axis=cell_axis,
        )
        phase, island_id, entity_id, placeholder_displaced_material = self._normalize_cell_runtime_arrays(
            material_id,
            phase,
            island_id,
            entity_id,
            placeholder_displaced_material,
        )
        runtime_payload = self._capture_page_stripe_island_runtime(
            island_id
        )
        runtime_payload["entity_placeholder_entity_id"] = self._capture_page_stripe_entity_placeholder_runtime(
            update,
            stripe_axis=cell_axis,
        )
        payload = {
            "meta": {
                "axis": update.axis,
                "world_start": update.world_start,
                "world_end": update.world_end,
                "buffer_start": update.buffer_start,
                "buffer_end": update.buffer_end,
                "kind": update.kind,
                "cross_world_start": update.cross_world_start,
                "cross_world_end": update.cross_world_end,
            },
            "cell": {
                "material_id": material_id,
                "phase": phase,
                "cell_flags": self._capture_stripe_array(self.cell_flags, update, stripe_axis=cell_axis),
                "velocity": self._capture_stripe_array(self.velocity, update, stripe_axis=cell_axis),
                "cell_temperature": self._capture_stripe_array(self.cell_temperature, update, stripe_axis=cell_axis),
                "timer_pack": self._capture_stripe_array(self.timer_pack, update, stripe_axis=cell_axis),
                "integrity": self._capture_stripe_array(self.integrity, update, stripe_axis=cell_axis),
                "island_id": island_id,
                "entity_id": entity_id,
                "placeholder_displaced_material": placeholder_displaced_material,
                "collapse_delay_pending": self._capture_stripe_array(
                    self.collapse_delay_pending.astype(np.uint8),
                    update,
                    stripe_axis=cell_axis,
                ),
                "visible_illumination": self._capture_stripe_array(
                    self.visible_illumination,
                    update,
                    stripe_axis=cell_axis,
                ),
                "cell_optical_dose": self._capture_stripe_array(
                    self.cell_optical_dose,
                    update,
                    stripe_axis=cell_dose_axis,
                ),
            },
            "runtime": runtime_payload,
            "gas": {
                "ambient_temperature": self._capture_stripe_array(
                    self.ambient_temperature,
                    update,
                    stripe_axis=1 if update.axis == "x" else 0,
                    ranges=gas_ranges,
                ),
                "flow_velocity": self._capture_stripe_array(
                    self.flow_velocity,
                    update,
                    stripe_axis=1 if update.axis == "x" else 0,
                    ranges=gas_ranges,
                ),
                "pressure_ping": self._capture_stripe_array(
                    self.pressure_ping,
                    update,
                    stripe_axis=1 if update.axis == "x" else 0,
                    ranges=gas_ranges,
                ),
                "gas_concentration": self._capture_stripe_array(
                    self.gas_concentration,
                    update,
                    stripe_axis=2 if update.axis == "x" else 1,
                    ranges=gas_ranges,
                ),
                "gas_optical_dose": self._capture_stripe_array(
                    self.gas_optical_dose,
                    update,
                    stripe_axis=2 if update.axis == "x" else 1,
                    ranges=gas_ranges,
                ),
            },
        }
        return payload

    def apply_page_stripe(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
        *,
        immediate: bool = False,
    ) -> None:
        update = self._contextualize_page_stripe_update(update)
        payload = self._coerce_page_stripe_payload(payload)
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            self.bridge_frame_paging_updates.append(PageStripeUpdate(**asdict(update)))
            self._apply_page_stripe(update, payload)
            self._record_bridge_page_stripe(update, payload)
            return
        self.queue_command(
            "apply_page_stripe",
            update=asdict(update),
            payload=payload,
        )

    def store_page_stripe(self, update: PageStripeUpdate, payload: dict[str, Any]) -> dict[str, Any]:
        update = self._contextualize_page_stripe_update(update)
        normalized_payload = self._coerce_page_stripe_payload(payload)
        self.page_store.save(update, normalized_payload)
        stored_payload = self.page_store.load(update)
        assert stored_payload is not None
        return self._coerce_page_stripe_payload(stored_payload)

    def capture_page_stripe_to_store(self, update: PageStripeUpdate) -> dict[str, Any]:
        return self.store_page_stripe(update, self.capture_page_stripe(update))

    def load_page_stripe(self, update: PageStripeUpdate) -> dict[str, Any] | None:
        update = self._contextualize_page_stripe_update(update)
        payload = self.page_store.load(update)
        if payload is None:
            return None
        return self._coerce_page_stripe_payload(payload)

    def apply_stored_page_stripe(
        self,
        update: PageStripeUpdate,
        *,
        immediate: bool = False,
    ) -> dict[str, Any] | None:
        payload = self.load_page_stripe(update)
        if payload is None:
            return None
        self.apply_page_stripe(update, payload, immediate=immediate)
        return payload

    def page_store_has_stripe(self, update: PageStripeUpdate) -> bool:
        update = self._contextualize_page_stripe_update(update)
        return bool(self.page_store.has(update))

    def list_page_store_stripe_keys(self) -> list[StoredStripeKey] | None:
        list_keys = getattr(self.page_store, "keys", None)
        if not callable(list_keys):
            return None
        return [self._coerce_page_store_key(key) for key in list_keys()]

    def export_page_store_entries(self) -> dict[str, Any]:
        keys = self.list_page_store_stripe_keys()
        entries: list[dict[str, Any]] = []
        if keys is not None:
            for key in keys:
                payload = self.page_store.load(self._page_store_key_lookup_update(key))
                if payload is None:
                    continue
                entries.append(
                    {
                        "key": self.serialize_page_store_key(key),
                        "payload": self.serialize_page_stripe_payload(self._coerce_page_stripe_payload(payload)),
                    }
                )
        return {
            "stored_stripes": int(self.page_store.stored_count()),
            "key_listing_supported": keys is not None,
            "entries": entries,
        }

    def import_page_store_entries(self, entries: Iterable[dict[str, Any]], *, clear: bool = False) -> dict[str, int]:
        cleared = 0
        if clear:
            cleared = self.clear_page_store()
        imported = 0
        for entry in entries:
            key = self._coerce_page_store_key(entry["key"])
            payload = self._coerce_page_stripe_payload(dict(entry["payload"]))
            self.page_store.save(self._page_store_key_lookup_update(key), payload)
            imported += 1
        return {
            "cleared": int(cleared),
            "imported": int(imported),
            "stored_stripes": int(self.page_store.stored_count()),
        }

    def clear_page_store(self) -> int:
        cleared = int(self.page_store.stored_count())
        self.page_store.clear()
        return cleared

    def serialize_page_store_state(self) -> dict[str, Any]:
        keys = self.list_page_store_stripe_keys()
        return {
            "stored_stripes": int(self.page_store.stored_count()),
            "key_listing_supported": keys is not None,
            "stripe_keys": []
            if keys is None
            else [self.serialize_page_store_key(key) for key in keys],
        }

    def _coerce_page_store_key(
        self,
        key: StoredStripeKey | PageStripeUpdate | dict[str, Any],
    ) -> StoredStripeKey:
        if isinstance(key, StoredStripeKey):
            return StoredStripeKey(
                axis=str(key.axis),
                world_start=int(key.world_start),
                world_end=int(key.world_end),
                cross_world_start=int(getattr(key, "cross_world_start", 0)),
                cross_world_end=int(getattr(key, "cross_world_end", 0)),
            )
        if isinstance(key, PageStripeUpdate):
            return StoredStripeKey(
                axis=str(key.axis),
                world_start=int(key.world_start),
                world_end=int(key.world_end),
                cross_world_start=0 if key.cross_world_start is None else int(key.cross_world_start),
                cross_world_end=0 if key.cross_world_end is None else int(key.cross_world_end),
            )
        payload = dict(key)
        return StoredStripeKey(
            axis=str(payload["axis"]),
            world_start=int(payload["world_start"]),
            world_end=int(payload["world_end"]),
            cross_world_start=int(payload.get("cross_world_start", 0)),
            cross_world_end=int(payload.get("cross_world_end", 0)),
        )

    @staticmethod
    def _page_store_key_lookup_update(key: StoredStripeKey) -> PageStripeUpdate:
        return PageStripeUpdate(
            axis=str(key.axis),
            world_start=int(key.world_start),
            world_end=int(key.world_end),
            buffer_start=0,
            buffer_end=max(1, int(key.world_end) - int(key.world_start)),
            kind="load",
            cross_world_start=int(getattr(key, "cross_world_start", 0)),
            cross_world_end=int(getattr(key, "cross_world_end", 0)),
        )

    def _coerce_page_stripe_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        cell_payload = dict(payload["cell"])
        gas_payload = dict(payload["gas"])
        runtime_payload = None if payload.get("runtime") is None else dict(payload["runtime"])
        gas_concentration = np.asarray(
            gas_payload["gas_concentration"],
            dtype=self.gas_concentration.dtype,
        ).copy()
        np.maximum(gas_concentration, 0.0, out=gas_concentration)
        if runtime_payload is not None and "entity_placeholder_entity_id" in runtime_payload:
            runtime_payload["entity_placeholder_entity_id"] = np.asarray(
                runtime_payload["entity_placeholder_entity_id"],
                dtype=np.int32,
            )
        if runtime_payload is not None:
            if "island_ids" in runtime_payload:
                runtime_payload["island_ids"] = np.asarray(runtime_payload["island_ids"], dtype=np.int32)
            if "island_velocity" in runtime_payload:
                runtime_payload["island_velocity"] = np.asarray(runtime_payload["island_velocity"], dtype=np.float32)
            if "island_subcell_offset" in runtime_payload:
                runtime_payload["island_subcell_offset"] = np.asarray(
                    runtime_payload["island_subcell_offset"],
                    dtype=np.float32,
                )
        return {
            "meta": dict(payload["meta"]),
            "cell": {
                "material_id": np.asarray(cell_payload["material_id"], dtype=self.material_id.dtype),
                "phase": np.asarray(cell_payload["phase"], dtype=self.phase.dtype),
                "cell_flags": np.asarray(cell_payload["cell_flags"], dtype=self.cell_flags.dtype),
                "velocity": np.asarray(cell_payload["velocity"], dtype=self.velocity.dtype),
                "cell_temperature": np.asarray(cell_payload["cell_temperature"], dtype=self.cell_temperature.dtype),
                "timer_pack": np.asarray(cell_payload["timer_pack"], dtype=self.timer_pack.dtype),
                "integrity": np.asarray(cell_payload["integrity"], dtype=self.integrity.dtype),
                "island_id": np.asarray(cell_payload["island_id"], dtype=self.island_id.dtype),
                "entity_id": np.asarray(cell_payload["entity_id"], dtype=self.entity_id.dtype),
                "placeholder_displaced_material": np.asarray(
                    cell_payload["placeholder_displaced_material"],
                    dtype=self.placeholder_displaced_material.dtype,
                ),
                "collapse_delay_pending": np.asarray(
                    cell_payload["collapse_delay_pending"],
                    dtype=np.uint8,
                ),
                "visible_illumination": np.asarray(
                    cell_payload["visible_illumination"],
                    dtype=self.visible_illumination.dtype,
                ),
                "cell_optical_dose": np.asarray(
                    cell_payload["cell_optical_dose"],
                    dtype=self.cell_optical_dose.dtype,
                ),
            },
            "runtime": runtime_payload,
            "gas": {
                "ambient_temperature": np.asarray(
                    gas_payload["ambient_temperature"],
                    dtype=self.ambient_temperature.dtype,
                ),
                "flow_velocity": np.asarray(gas_payload["flow_velocity"], dtype=self.flow_velocity.dtype),
                "pressure_ping": np.asarray(gas_payload["pressure_ping"], dtype=self.pressure_ping.dtype),
                "gas_concentration": gas_concentration,
                "gas_optical_dose": np.asarray(
                    gas_payload["gas_optical_dose"],
                    dtype=self.gas_optical_dose.dtype,
                ),
            },
        }

    def sync_entity_placeholders(
        self,
        placeholders: list[EntityPlaceholder],
        *,
        immediate: bool = False,
    ) -> None:
        placeholders = [self._public_entity_placeholder_input(placeholder) for placeholder in placeholders]
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            self._sync_entity_placeholders(
                [self._frame_entity_placeholder_input(placeholder) for placeholder in placeholders]
            )
            return
        self.queue_command(
            "sync_entity_placeholders",
            placeholders=[asdict(placeholder) for placeholder in placeholders],
        )

    def sync_entity_states(
        self,
        entities: list[EntityState | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        entities = [self._public_entity_state_input(entity) for entity in entities]
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            placeholders, _ = self._sync_entity_states(
                [self._frame_entity_state_input(entity) for entity in entities]
            )
            self._sync_entity_placeholders(placeholders)
            return
        self.queue_command(
            "sync_entity_states",
            entities=[asdict(entity) for entity in entities],
        )

    def patch_entity_states(
        self,
        patches: list[EntityStatePatch | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        patches = [self._public_entity_state_patch_input(patch) for patch in patches]
        if immediate:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            self._patch_entity_states(
                [self._frame_entity_state_patch_input(patch) for patch in patches]
            )
            return
        self.queue_command(
            "patch_entity_states",
            patches=[asdict(patch) for patch in patches],
        )

    def sync_entity_observation_specs(
        self,
        observations: list[EntityObservationSpec | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        observations = [self._coerce_entity_observation_spec(observation) for observation in observations]
        if immediate:
            self._sync_entity_observation_specs(observations)
            return
        self.queue_command(
            "sync_entity_observation_specs",
            observations=[asdict(observation) for observation in observations],
        )

    def set_force_sources(
        self,
        force_sources: list[ForceSource | dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        force_sources = [self._public_force_source_input(force_source) for force_source in force_sources]
        if immediate:
            self._sync_force_sources(
                [self._normalize_runtime_force_source(force_source) for force_source in force_sources]
            )
            return
        self.queue_command(
            "set_force_sources",
            force_sources=[
                {
                    "x": float(force_source.world_x),
                    "y": float(force_source.world_y),
                    "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
                    "radius": float(force_source.radius),
                    "strength": float(force_source.strength),
                    "lifetime": float(force_source.lifetime),
                }
                for force_source in force_sources
            ],
        )

    def set_emitters(
        self,
        emitters: list[dict[str, Any]],
        *,
        immediate: bool = False,
    ) -> None:
        emitters = [self._coerce_emitter(emitter) for emitter in emitters]
        if immediate:
            normalized_emitters = [
                {
                    **dict(emitter),
                    "origin": self._world_to_buffer_clamped(
                        int(emitter["world_origin"][0]),
                        int(emitter["world_origin"][1]),
                    ),
                }
                for emitter in emitters
            ]
            self._sync_persistent_emitters(normalized_emitters)
            return
        self.queue_command(
            "set_emitters",
            emitters=[
                {
                    "x": int(emitter["origin"][0]),
                    "y": int(emitter["origin"][1]),
                    "light_type": str(emitter["light_type"]),
                    "direction": list(emitter["direction"]),
                    "spread": float(emitter["spread"]),
                    "strength": float(emitter["strength"]),
                    "radius": int(emitter["range_cells"]),
                }
                for emitter in emitters
            ],
        )

    def patch_material(self, name: str, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command(
                "patch_material",
                name=str(self._canonical_material_input_name(name)),
                fields=self._normalize_material_patch_fields(fields),
            )
            return
        material_id = self._resolve_sanctioned_material_id(name)
        if material_id <= 0:
            raise KeyError(name)
        material = self._shadow_material_def(material_id)
        if material is None:
            raise KeyError(name)
        patch_fields = dict(fields)
        patch_fields.setdefault("name", self._shadow_material_name(material_id))
        patch_fields.setdefault("material_id", int(material_id))
        updated = self._coerce_material_def(asdict(replace(material, **patch_fields)))
        self.update_material_table([updated])

    def patch_light(self, name: str, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command("patch_light", name=name, fields=fields)
            return
        light_id = self._resolve_sanctioned_light_id(name)
        if light_id < 0:
            raise KeyError(name)
        light = self._shadow_light_type_def(light_id)
        if light is None:
            raise KeyError(name)
        patch_fields = dict(fields)
        patch_fields.setdefault("name", self._shadow_light_name(light_id))
        patch_fields.setdefault("light_type_id", int(light_id))
        updated = self._coerce_light_type_def(asdict(replace(light, **patch_fields)))
        self.update_light_type_table([updated])

    def patch_gas(self, name: str, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command(
                "patch_gas",
                name=name,
                fields=self._normalize_gas_patch_fields(fields),
            )
            return
        species_id = self._resolve_sanctioned_gas_id(name)
        if species_id < 0:
            raise KeyError(name)
        gas = self._shadow_gas_species_def(species_id)
        if gas is None:
            raise KeyError(name)
        patch_fields = dict(fields)
        patch_fields.setdefault("name", self._shadow_gas_name(species_id))
        patch_fields.setdefault("species_id", int(species_id))
        updated = self._coerce_gas_species_def(asdict(replace(gas, **patch_fields)))
        self.update_gas_species_table([updated])

    def patch_material_optics(
        self,
        material_name: str,
        light_type: str,
        *,
        immediate: bool = True,
        **fields: Any,
    ) -> None:
        if not immediate:
            self.queue_command(
                "patch_material_optics",
                material_name=str(self._canonical_material_input_name(material_name)),
                light_type=light_type,
                fields=self._normalize_material_optics_patch_fields(fields),
            )
            return
        material_id = self._resolve_sanctioned_material_id(material_name)
        if material_id <= 0:
            raise KeyError(material_name)
        light_id = self._resolve_sanctioned_light_id(light_type)
        if light_id < 0:
            raise KeyError(light_type)
        canonical_material_name = self._shadow_material_name(material_id)
        canonical_light_type = self._shadow_light_name(light_id)
        if canonical_material_name is None or canonical_light_type is None:
            raise KeyError((material_name, light_type))
        optics = self._shadow_material_optics_def(canonical_material_name, canonical_light_type)
        if optics is None:
            raise KeyError((material_name, light_type))
        patch_fields = dict(fields)
        patch_fields.setdefault("material_name", canonical_material_name)
        patch_fields.setdefault("light_type", canonical_light_type)
        updated = self._coerce_material_optics_def(asdict(replace(optics, **patch_fields)))
        self.update_material_optics_table([updated])

    def patch_reaction_action(self, index: int, *, immediate: bool = True, **fields: Any) -> None:
        if not immediate:
            self.queue_command(
                "patch_reaction_action",
                index=index,
                fields=self._normalize_reaction_action_patch_fields(fields),
            )
            return
        if index <= 0:
            raise ValueError("reaction action 0 is reserved")
        if index >= len(self.rulebook.reaction_actions):
            raise IndexError(index)
        action = self._shadow_reaction_action(index)
        if action is None:
            raise IndexError(index)
        updated = self._coerce_reaction_action(asdict(replace(action, **fields)))
        reactions_payload = self._shadow_reaction_payload()
        if index == 0:
            self.rulebook.reaction_actions = [updated] + [
                self._coerce_reaction_action(item) for item in reactions_payload["actions"]
            ]
            self._validate_reaction_payload(reactions_payload)
            self._set_stable_shadow_payload("reactions", reactions_payload)
            self.bridge.upload_table("reactions", reactions_payload)
            self.bridge.sync_rule_tables(self)
            return
        reactions_payload["actions"][index - 1] = asdict(updated)
        self._validate_reaction_payload(reactions_payload)
        self.rulebook.reaction_actions = [self.rulebook.reaction_actions[0]] + [
            self._coerce_reaction_action(item) for item in reactions_payload["actions"]
        ]
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)

    def patch_reaction_rule(
        self,
        rule_set: str,
        index: int,
        *,
        immediate: bool = True,
        **fields: Any,
    ) -> None:
        rule_set = str(rule_set)
        if rule_set not in REACTION_RULE_SET_NAMES:
            raise KeyError(rule_set)
        if not immediate:
            self.queue_command(
                "patch_reaction_rule",
                rule_set=rule_set,
                index=index,
                fields=self._normalize_reaction_rule_patch_fields(fields),
            )
            return
        rule = self._shadow_reaction_rule(rule_set, index)
        if rule is None:
            raise IndexError(index)
        if rule_set == "self_rules":
            updated = self._coerce_self_reaction_rule(asdict(replace(rule, **fields)))
        else:
            updated = self._coerce_pair_reaction_rule(asdict(replace(rule, **fields)))
        reactions_payload = self._shadow_reaction_payload()
        reactions_payload["rules"][rule_set][index] = asdict(updated)
        self._validate_reaction_payload(reactions_payload)
        self._set_reaction_rule_list(rule_set, reactions_payload["rules"][rule_set])
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)

    def delete_reaction_action(self, index: int, *, immediate: bool = True) -> None:
        if not immediate:
            self.queue_command("delete_reaction_action", index=index)
            return
        if index <= 0:
            raise ValueError("reaction action 0 is reserved")
        if index >= len(self.rulebook.reaction_actions):
            raise IndexError(index)
        reactions_payload = self._shadow_reaction_payload()
        actions_payload = reactions_payload["actions"]
        if index > len(actions_payload):
            raise IndexError(index)
        materials_payload = self._shadow_material_payload()
        del actions_payload[index - 1]
        self._remap_reaction_payload_result_actions(reactions_payload["rules"], deleted_action_index=index)
        self._remap_material_payload_reaction_slots(materials_payload, deleted_action_index=index)
        self.rulebook.reaction_actions = [self.rulebook.reaction_actions[0]] + [
            self._coerce_reaction_action(item) for item in actions_payload
        ]
        self._set_reaction_rules_payload(reactions_payload["rules"])
        self.rulebook.materials_by_name.clear()
        self.rulebook.materials_by_id.clear()
        self.rulebook.update_materials(self._coerce_material_def(item) for item in materials_payload)
        self.tag_bits_by_name = deepcopy(self.rulebook.tag_bits)
        self._rebuild_material_property_arrays()
        self._set_stable_shadow_payload("materials", materials_payload)
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("materials", materials_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)
        self.bridge.ensure_world_resources(self)

    def delete_reaction_rule(self, rule_set: str, index: int, *, immediate: bool = True) -> None:
        rule_set = str(rule_set)
        if rule_set not in REACTION_RULE_SET_NAMES:
            raise KeyError(rule_set)
        if not immediate:
            self.queue_command("delete_reaction_rule", rule_set=rule_set, index=index)
            return
        rules_list = self._reaction_rule_list(rule_set)
        if index < 0 or index >= len(rules_list):
            raise IndexError(index)
        reactions_payload = self._shadow_reaction_payload()
        del reactions_payload["rules"][rule_set][index]
        self._set_reaction_rule_list(rule_set, reactions_payload["rules"][rule_set])
        self._set_stable_shadow_payload("reactions", reactions_payload)
        self.bridge.upload_table("reactions", reactions_payload)
        self.bridge.sync_rule_tables(self)

    def step(self, dt: float = 1.0 / 60.0, substeps: int = 1) -> None:
        for _ in range(max(1, substeps)):
            frame_input = self.pending_frame_inputs.popleft() if self.pending_frame_inputs else None
            output = self._step_once(dt, frame_input=frame_input, capture_output=frame_input is not None)
            if output is not None:
                self.completed_frame_outputs.append(output)

    def simulation_backend_report(self) -> dict[str, Any]:
        ctx = self.bridge.ctx
        gpu_available = bool(self.bridge.enabled and ctx is not None and getattr(ctx, "version_code", 0) >= 430)
        ctx_info = getattr(ctx, "info", {}) if ctx is not None else {}
        backends = {
            "collapse": str(self.collapse_solver.last_backend),
            "gas": str(self.gas_solver.last_backend),
            "heat": str(self.heat_solver.last_backend),
            "reactions": str(self.reaction_solver.last_runtime_backend),
            "motion": str(self.motion_solver.last_backend),
            "liquid": str(self.liquid_solver.last_backend),
            "placeholder": str(self.placeholder_pipeline.last_backend),
            "page_stripe": str(self.page_stripe_pipeline.last_backend),
            "world_commands": str(self.grid_command_pipeline.last_backend),
            "optics": str(self.optics_solver.last_backend),
        }
        non_gpu = {name: backend for name, backend in backends.items() if backend not in {"gpu", "idle"}}
        return {
            "simulation_backend": self.simulation_backend,
            "gpu_available": gpu_available,
            "renderer": str(ctx_info.get("GL_RENDERER", "")),
            "vendor": str(ctx_info.get("GL_VENDOR", "")),
            "opengl_version": str(ctx_info.get("GL_VERSION", "")),
            "gpu_realtime_budget": {
                "enabled": bool(self.gpu_realtime_budget_enabled),
                "active": bool(self._gpu_realtime_budget_active()),
                "cell_threshold": int(self.gpu_realtime_budget_cell_threshold),
                "skipped_stages": list(self.last_skipped_gpu_stages),
            },
            "backends": backends,
            "non_gpu_backends": non_gpu,
            "strict_gpu_ready": gpu_available and not non_gpu,
        }

    def poll_readbacks(self, request_id: int | None = None) -> ReadbackResult | None:
        if request_id is None:
            if not self.completed_readbacks:
                return None
            return self.completed_readbacks.popleft()
        for index, result in enumerate(self.completed_readbacks):
            if result.request.request_id == int(request_id):
                del self.completed_readbacks[index]
                return result
        return None

    def poll_all_readbacks(self, *, current_frame_id: int | None = None) -> list[ReadbackResult]:
        results: list[ReadbackResult] = []
        if current_frame_id is not None:
            self._collect_ready_readbacks(current_frame_id)
        while self.completed_readbacks:
            results.append(self.completed_readbacks.popleft())
        return results

    def consume_entity_observation_results(
        self,
        *,
        current_frame_id: int | None = None,
    ) -> dict[str, Any]:
        consumed_readbacks = self.poll_all_readbacks(current_frame_id=current_frame_id)
        observations = self._collect_observations(consumed_readbacks)
        entity_feedback = self._collect_entity_feedback(consumed_readbacks)
        frame_id = self.frame_id if current_frame_id is None else int(current_frame_id)
        return self._store_entity_observation_consume_snapshot(
            frame_id=frame_id,
            consumed_readbacks=consumed_readbacks,
            observations=observations,
            entity_feedback=entity_feedback,
        )

    def run_entity_controller_turn(
        self,
        *,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if controller_state_provided or controller_state is not None:
            self.controller_state_snapshot = self._coerce_json_value(controller_state)
        normalized_controller_state = deepcopy(self.controller_state_snapshot)
        consumed = self.consume_entity_observation_results()
        paging_updates: list[PageStripeUpdate] = []
        if focus_center is not None:
            paging_updates = self.advance_paging(int(focus_center[0]), int(focus_center[1]), immediate=True)
        if entities is not None:
            self.sync_entity_states(entities, immediate=True)
        if patches is not None:
            self.patch_entity_states(patches, immediate=True)
        if entity_placeholders is not None:
            placeholder_inputs = [
                self._coerce_entity_placeholder(placeholder)
                for placeholder in entity_placeholders
            ]
            if entities is not None:
                entity_placeholder_inputs, _ = self._frame_entities_to_placeholders_and_observations(list(self.entity_states.values()))
                placeholder_inputs = entity_placeholder_inputs + placeholder_inputs
            self.sync_entity_placeholders(placeholder_inputs, immediate=True)
        if observation_specs is not None:
            self.sync_entity_observation_specs(observation_specs, immediate=True)
        if force_sources is not None:
            self.set_force_sources(force_sources, immediate=True)
        if emitters is not None:
            self.set_emitters(emitters, immediate=True)
        resolved_targets = self._resolve_target_queries(
            [self._coerce_target_query(query) for query in (target_queries or [])]
        )
        resolved_change_intents, generated_commands = self._resolve_change_intents(
            [self._coerce_change_intent(intent) for intent in (change_intents or [])],
            resolved_targets,
        )
        resolved_carrier_intents, generated_carrier_commands = self._resolve_carrier_intents(
            [self._coerce_carrier_intent(intent) for intent in (carrier_intents or [])],
            resolved_targets,
        )
        entity_observation_targets = self._runtime_entities_to_immediate_observation_targets(
            list(self.entity_states.values())
        )
        queued_observation_requests: list[ReadbackRequest] = []
        all_observation_targets = entity_observation_targets + [
            self._coerce_observation_target(target) for target in (observation_targets or [])
        ]
        if all_observation_targets:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            queued_observation_requests = self._build_observation_requests(
                all_observation_targets,
                resolved_targets,
            )
            queued_observation_requests = [
                self._assign_readback_request_id(request) for request in queued_observation_requests
            ]
            self.pending_readbacks.extend(queued_observation_requests)
            self.bridge_frame_readback_requests.extend(replace(request) for request in queued_observation_requests)
        queued_readback_requests: list[ReadbackRequest] = []
        if readback_requests:
            if not self._bridge_inputs_prepared:
                self._prepare_bridge_frame_inputs()
            queued_readback_requests = self._resolve_readback_requests(
                [self._coerce_readback_request(request) for request in readback_requests],
                resolved_targets,
            )
            queued_readback_requests = [
                self._assign_readback_request_id(request) for request in queued_readback_requests
            ]
            self.pending_readbacks.extend(queued_readback_requests)
            self.bridge_frame_readback_requests.extend(replace(request) for request in queued_readback_requests)
        resolved_commands = (
            generated_commands
            + generated_carrier_commands
            + self._resolve_targeted_commands(
                [self._coerce_world_command(command) for command in (commands or [])],
                resolved_targets,
            )
        )
        for command in resolved_commands:
            self.queue_command(command.kind, **command.payload)
        return {
            "frame_id": int(self.frame_id),
            "controller_state": normalized_controller_state,
            "consumed": consumed,
            "paging_updates": [asdict(update) for update in paging_updates],
            "resolved_targets": {
                query_id: self.serialize_resolved_target(target)
                for query_id, target in resolved_targets.items()
            },
            "resolved_change_intents": {
                intent_id: self.serialize_resolved_change_intent(self._public_resolved_change_intent(intent))
                for intent_id, intent in resolved_change_intents.items()
            },
            "resolved_carrier_intents": {
                intent_id: self.serialize_resolved_carrier_intent(self._public_resolved_carrier_intent(intent))
                for intent_id, intent in resolved_carrier_intents.items()
            },
            "resolved_commands": [self.serialize_world_command(command) for command in resolved_commands],
            "observation_requests": [
                self.serialize_readback_request(request) for request in queued_observation_requests
            ],
            "readback_requests": [
                self.serialize_readback_request(request) for request in queued_readback_requests
            ],
            "queued_observations": len(queued_observation_requests),
            "queued_readbacks": len(queued_readback_requests),
            "queued_commands": len(resolved_commands),
            "entities": self.serialize_entity_states()["entities"],
            "placeholders": self._serialize_cpu_visible_entity_placeholders()["placeholders"],
            "observation_state": self.serialize_entity_observation_state(),
            "paging_state": self.serialize_paging_state(),
            "readback_state": self.serialize_readback_state(),
            "force_sources": self.serialize_force_sources(),
            "emitters": self.serialize_emitters(),
            "pending_commands": self.serialize_pending_commands(),
        }

    def set_controller_state(self, controller_state: Any = None) -> dict[str, Any]:
        self.controller_state_snapshot = self._coerce_json_value(controller_state)
        return self.serialize_controller_state()

    def serialize_controller_state(self) -> dict[str, Any]:
        return {"controller_state": deepcopy(self.controller_state_snapshot)}

    def _build_preview_controller_turn_entities(
        self,
        *,
        entities: list[EntityState | dict[str, Any]] | None,
        patches: list[EntityStatePatch | dict[str, Any]] | None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None,
    ) -> list[EntityState] | None:
        if entities is None and patches is None and observation_specs is None:
            return None
        next_entities = {
            entity.entity_id: entity
            for entity in (
                [self._controller_turn_entity_input(entity) for entity in entities]
                if entities is not None
                else [
                    replace(entity, world_x=None, world_y=None)
                    for _, entity in sorted(self.entity_states.items())
                ]
            )
        }
        if patches is not None:
            for patch in [self._coerce_entity_state_patch(patch) for patch in patches]:
                entity = next_entities.get(patch.entity_id)
                if entity is None:
                    raise KeyError(patch.entity_id)
                patch_fields = {name: value for name, value in patch.fields.items() if not name.startswith("_")}
                next_entity = replace(entity, **dict(patch_fields))
                if "_world_x" in patch.fields or "_world_y" in patch.fields:
                    next_entity = replace(
                        next_entity,
                        world_x=int(patch.fields.get("_world_x", entity.world_x if entity.world_x is not None else entity.x)),
                        world_y=int(patch.fields.get("_world_y", entity.world_y if entity.world_y is not None else entity.y)),
                    )
                elif "x" in patch_fields or "y" in patch_fields:
                    next_entity = replace(next_entity, world_x=None, world_y=None)
                next_entities[patch.entity_id] = self._coerce_entity_state(next_entity)
        if observation_specs is not None:
            observation_by_entity_id = {
                observation.entity_id: observation
                for observation in [self._coerce_entity_observation_spec(spec) for spec in observation_specs]
            }
            next_entities = {
                entity_id: replace(
                    entity,
                    observe_channels=observation.observe_channels if observation is not None else (),
                    observe_pad_cells=int(observation.observe_pad_cells) if observation is not None else 0,
                    observe_width=None if observation is None else observation.observe_width,
                    observe_height=None if observation is None else observation.observe_height,
                    observe_label=None if observation is None else observation.observe_label,
                )
                for entity_id, entity in next_entities.items()
                for observation in [observation_by_entity_id.get(entity_id)]
            }
        return [next_entities[entity_id] for entity_id in sorted(next_entities)]

    def _preview_consume_entity_observation_results(self) -> dict[str, Any]:
        saved_completed_readbacks = deepcopy(self.completed_readbacks)
        saved_last_snapshot = deepcopy(self.last_entity_observation_consume_snapshot)
        try:
            return self.consume_entity_observation_results()
        finally:
            self.completed_readbacks = saved_completed_readbacks
            self.last_entity_observation_consume_snapshot = saved_last_snapshot

    def controller_turn_to_frame_input(
        self,
        *,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> WorldFrameInput:
        preview_entities = self._build_preview_controller_turn_entities(
            entities=entities,
            patches=patches,
            observation_specs=observation_specs,
        )
        normalized_controller_state_provided = bool(controller_state_provided or controller_state is not None)
        return WorldFrameInput(
            focus_center=focus_center,
            controller_state=(
                self._coerce_json_value(controller_state)
                if normalized_controller_state_provided
                else None
            ),
            controller_state_provided=normalized_controller_state_provided,
            entities=preview_entities,
            entity_placeholders=None
            if entity_placeholders is None
            else [self._public_entity_placeholder_input(placeholder) for placeholder in entity_placeholders],
            force_sources=[]
            if force_sources == []
            else None
            if force_sources is None
            else [self._public_force_source_input(force_source) for force_source in force_sources],
            emitters=[]
            if emitters == []
            else None
            if emitters is None
            else [self._coerce_emitter(emitter) for emitter in emitters],
            target_queries=[self._coerce_target_query(query) for query in (target_queries or [])],
            change_intents=[self._coerce_change_intent(intent) for intent in (change_intents or [])],
            carrier_intents=[self._coerce_carrier_intent(intent) for intent in (carrier_intents or [])],
            observation_targets=[self._coerce_observation_target(target) for target in (observation_targets or [])],
            readback_requests=[self._coerce_readback_request(request) for request in (readback_requests or [])],
            commands=[self._coerce_world_command(command) for command in (commands or [])],
        )

    def preview_entity_controller_turn(
        self,
        *,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
        reserved_readback_request_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        frame_input = self.controller_turn_to_frame_input(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        normalized_controller_state = (
            deepcopy(frame_input.controller_state)
            if frame_input.controller_state_provided
            else deepcopy(self.controller_state_snapshot)
        )
        consumed = self._preview_consume_entity_observation_results()
        force_sources_payload = (
            self.serialize_force_sources()
            if force_sources is None
            else [self._serialize_force_source_record(force_source) for force_source in frame_input.force_sources or []]
        )
        emitters_payload = (
            self.serialize_emitters()
            if emitters is None
            else {
                "persistent_emitters": [self._serialize_emitter_record(emitter) for emitter in frame_input.emitters or []],
                "queued_emitters": [],
            }
        )
        pending_commands_payload = self.serialize_pending_commands()
        pending_commands = {
            "pending": int(pending_commands_payload["pending"]),
            "commands": list(pending_commands_payload["commands"]),
        }

        saved_paging = deepcopy(self.paging)
        saved_preview_runtime = self._snapshot_preview_runtime_state()
        saved_entity_states = dict(self.entity_states)
        saved_entity_placeholders = {entity_id: set(cells) for entity_id, cells in self.entity_placeholders.items()}
        saved_blocked_cells = None if self._resolver_blocked_cells is None else set(self._resolver_blocked_cells)
        saved_released_cells = None if self._resolver_released_cells is None else set(self._resolver_released_cells)
        try:
            (
                paging_updates,
                preview_page_stripes,
                entity_observation_targets,
                placeholder_inputs,
                placeholder_count,
            ) = self._prepare_preview_frame_context(frame_input)
            resolved_targets = self._resolve_target_queries(frame_input.target_queries)
            resolved_change_intents, generated_commands = self._resolve_change_intents(frame_input.change_intents, resolved_targets)
            resolved_carrier_intents, generated_carrier_commands = self._resolve_carrier_intents(
                frame_input.carrier_intents,
                resolved_targets,
            )
            observation_pairs = self._build_observation_request_pairs(
                entity_observation_targets + frame_input.observation_targets,
                resolved_targets,
            )
            observation_requests, next_preview_request_id = self._assign_preview_readback_request_ids(
                [request for _, request in observation_pairs]
            )
            observation_pairs = [
                (target, request)
                for (target, _), request in zip(observation_pairs, observation_requests, strict=False)
            ]
            resolved_commands = (
                generated_commands
                + generated_carrier_commands
                + self._resolve_targeted_commands(frame_input.commands, resolved_targets)
            )
            readback_request_plan, _ = self._assign_preview_readback_request_ids(
                self._resolve_readback_requests(frame_input.readback_requests, resolved_targets),
                next_request_id=next_preview_request_id,
            )
            pending_commands["pending"] += len(resolved_commands)
            pending_commands["commands"].extend(
                self.serialize_world_command(command)
                for command in resolved_commands
            )
            bridge_frame_snapshot = self._serialize_preview_bridge_frame_snapshot(
                current_entity_placeholders=saved_entity_placeholders,
                resolved_commands=resolved_commands,
                observation_requests=observation_requests,
                readback_requests=readback_request_plan,
                placeholder_inputs=placeholder_inputs,
                paging_updates=paging_updates,
                page_stripes=preview_page_stripes,
                reserved_readback_request_ids=reserved_readback_request_ids,
            )
            return {
                "frame_id": int(self.frame_id),
                "controller_state": normalized_controller_state,
                "consumed": consumed,
                "paging_updates": [asdict(update) for update in paging_updates],
                "resolved_targets": {
                    query_id: self.serialize_resolved_target(target)
                    for query_id, target in resolved_targets.items()
                },
                "resolved_change_intents": {
                    intent_id: self.serialize_resolved_change_intent(self._public_resolved_change_intent(intent))
                    for intent_id, intent in resolved_change_intents.items()
                },
                "resolved_carrier_intents": {
                    intent_id: self.serialize_resolved_carrier_intent(self._public_resolved_carrier_intent(intent))
                    for intent_id, intent in resolved_carrier_intents.items()
                },
                "resolved_commands": [self.serialize_world_command(command) for command in resolved_commands],
                "observation_requests": [
                    self.serialize_readback_request(request) for request in observation_requests
                ],
                "observation_plans": [
                    self._serialize_observation_plan_for_target_request(target, request)
                    for target, request in observation_pairs
                ],
                "readback_requests": [
                    self.serialize_readback_request(request) for request in readback_request_plan
                ],
                "readback_plans": self._serialize_readback_plans_for_requests(readback_request_plan),
                "bridge_frame_snapshot": bridge_frame_snapshot,
                "queued_observations": len(observation_requests),
                "queued_readbacks": len(readback_request_plan),
                "queued_commands": len(resolved_commands),
                "placeholder_count": int(placeholder_count),
                "entities": self.serialize_entity_states()["entities"],
                "placeholders": self._serialize_cpu_visible_entity_placeholders()["placeholders"],
                "observation_state": self.serialize_entity_observation_state(),
                "paging_state": self.serialize_paging_state(),
                "force_sources": force_sources_payload,
                "emitters": emitters_payload,
                "pending_commands": pending_commands,
            }
        finally:
            self._restore_preview_runtime_state(saved_preview_runtime)
            self.paging = saved_paging
            self.entity_states = saved_entity_states
            self.entity_placeholders = saved_entity_placeholders
            self._resolver_blocked_cells = saved_blocked_cells
            self._resolver_released_cells = saved_released_cells

    def submit_entity_controller_turn(
        self,
        *,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> int:
        frame_input = self.controller_turn_to_frame_input(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        return self.submit_frame_input(frame_input)

    def request_entity_controller_turn(
        self,
        *,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        submission_id = self.submit_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        pending_frame_input = self._pending_frame_input(submission_id)
        preview = self.preview_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=[
                replace(request)
                for request in pending_frame_input.readback_requests
            ],
            commands=commands,
            reserved_readback_request_ids=set(self._frame_readback_request_ids(pending_frame_input)),
        )
        return {
            "queued": True,
            "pending_frames": len(self.pending_frame_inputs),
            "submission_id": submission_id,
            "preview": preview,
        }

    def request_entity_controller_cycle(
        self,
        *,
        apply_turn: bool = True,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        preview = self.preview_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        if not apply_turn:
            return {
                "applied": False,
                "queued": False,
                "pending_frames": len(self.pending_frame_inputs),
                "submission_id": None,
                "preview": preview,
                "result": None,
            }
        submission_id = self.submit_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        pending_frame_input = self._pending_frame_input(submission_id)
        preview = self.preview_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=[
                replace(request)
                for request in pending_frame_input.readback_requests
            ],
            commands=commands,
            reserved_readback_request_ids=set(self._frame_readback_request_ids(pending_frame_input)),
        )
        return {
            "applied": True,
            "queued": True,
            "pending_frames": len(self.pending_frame_inputs),
            "submission_id": submission_id,
            "preview": preview,
            "result": None,
        }

    def run_entity_controller_cycle(
        self,
        *,
        apply_turn: bool = True,
        controller_state: Any = None,
        controller_state_provided: bool = False,
        focus_center: tuple[int, int] | None = None,
        entities: list[EntityState | dict[str, Any]] | None = None,
        entity_placeholders: list[EntityPlaceholder | dict[str, Any]] | None = None,
        patches: list[EntityStatePatch | dict[str, Any]] | None = None,
        observation_specs: list[EntityObservationSpec | dict[str, Any]] | None = None,
        force_sources: list[ForceSource | dict[str, Any]] | None = None,
        emitters: list[dict[str, Any]] | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
        change_intents: list[ChangeIntent | dict[str, Any]] | None = None,
        carrier_intents: list[CarrierIntent | dict[str, Any]] | None = None,
        observation_targets: list[ObservationTarget | dict[str, Any]] | None = None,
        readback_requests: list[ReadbackRequest | dict[str, Any]] | None = None,
        commands: list[WorldCommand | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        preview = self.preview_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        if not apply_turn:
            return {
                "applied": False,
                "preview": preview,
                "result": None,
            }
        result = self.run_entity_controller_turn(
            controller_state=controller_state,
            controller_state_provided=controller_state_provided,
            focus_center=focus_center,
            entities=entities,
            entity_placeholders=entity_placeholders,
            patches=patches,
            observation_specs=observation_specs,
            force_sources=force_sources,
            emitters=emitters,
            target_queries=target_queries,
            change_intents=change_intents,
            carrier_intents=carrier_intents,
            observation_targets=observation_targets,
            readback_requests=readback_requests,
            commands=commands,
        )
        return {
            "applied": True,
            "preview": preview,
            "result": result,
        }

    def run_cpu_frame(
        self,
        frame_input: WorldFrameInput | None = None,
        *,
        dt: float = 1.0 / 60.0,
        substeps: int = 1,
    ) -> WorldFrameOutput:
        output = self._step_once(dt, frame_input=frame_input, capture_output=True)
        assert output is not None
        for _ in range(max(1, substeps) - 1):
            self._step_once(dt, frame_input=None, capture_output=False)
            output.frame_id = self.frame_id
        return output

    def _step_once(
        self,
        dt: float,
        *,
        frame_input: WorldFrameInput | None,
        capture_output: bool,
    ) -> WorldFrameOutput | None:
        previous_frame_active = self._world_simulation_frame_active
        self._world_simulation_frame_active = True
        try:
            return self._step_once_impl(dt, frame_input=frame_input, capture_output=capture_output)
        finally:
            self._world_simulation_frame_active = previous_frame_active

    def _step_once_impl(
        self,
        dt: float,
        *,
        frame_input: WorldFrameInput | None,
        capture_output: bool,
    ) -> WorldFrameOutput | None:
        self.last_skipped_gpu_stages = []
        self.last_pass_profile = {"passes": [], "summary": {}, "skipped_stages": self.last_skipped_gpu_stages}
        if not self._bridge_inputs_prepared:
            self._prepare_bridge_frame_inputs()
        consumed_readbacks: list[ReadbackResult] = []
        resolved_targets: dict[str, ResolvedTarget] = {}
        resolved_change_intents: dict[str, ResolvedChangeIntent] = {}
        resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = {}
        observations: dict[int, ObservationResult] = {}
        entity_feedback: dict[int, EntityFeedback] = {}
        paging_updates: list[PageStripeUpdate] = []
        observation_plans: list[dict[str, Any]] = []
        readback_plans: list[dict[str, Any]] = []
        bridge_upload_snapshot: dict[str, Any] = {}
        bridge_frame_snapshot: dict[str, Any] = {}
        output_controller_state = deepcopy(self.controller_state_snapshot)
        queued_observations = 0
        queued_readbacks = 0
        queued_commands = 0
        placeholder_count = 0
        with self._profile_pass("readback"):
            self._collect_ready_readbacks(self.frame_id + 1)
            if capture_output:
                consumed_readbacks = self.poll_all_readbacks()
                observations = self._collect_observations(consumed_readbacks)
                entity_feedback = self._collect_entity_feedback(consumed_readbacks)
        with self._profile_pass("commands"):
            if frame_input is not None:
                (
                    output_controller_state,
                    paging_updates,
                    resolved_targets,
                    resolved_change_intents,
                    resolved_carrier_intents,
                    observation_plans,
                    readback_plans,
                    queued_observations,
                    queued_readbacks,
                    queued_commands,
                    placeholder_count,
                ) = self._apply_frame_input(frame_input)
            else:
                output_controller_state = deepcopy(self.controller_state_snapshot)

        self.frame_id += 1
        if capture_output:
            self._store_entity_observation_consume_snapshot(
                frame_id=self.frame_id,
                consumed_readbacks=consumed_readbacks,
                observations=observations,
                entity_feedback=entity_feedback,
            )
        with self._profile_pass("commands"):
            self._apply_commands()
        with self._profile_pass("pre_sync"):
            if self._needs_pre_simulation_bridge_sync(frame_input=frame_input):
                self._sync_pre_simulation_bridge_without_debug_upload()
                self._gpu_cpu_dirty_resources.clear()
        with self._profile_pass("commands"):
            persistent_observation_plans = self._queue_persistent_entity_observations()
            observation_plans.extend(persistent_observation_plans)
            queued_observations += len(persistent_observation_plans)
        if self.profile_passes_enabled:
            self.collapse_solver.gpu_pipeline.reset_pass_profile()
        collapse_pipeline = self.collapse_solver.gpu_pipeline
        with self._profile_pass("collapse"):
            if self._should_run_formal_collapse_this_frame():
                with collapse_pipeline._profile_pass(self, "dirty_tile_drain"):
                    self._drain_gpu_collapse_structure_dirty_tiles()
                self.collapse_solver.step(self)
            else:
                with collapse_pipeline._profile_pass(self, "scheduled_defer"):
                    pass
        collapse_profile = getattr(getattr(self.collapse_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(collapse_profile, dict):
            self.last_pass_profile["collapse"] = collapse_profile
        if self.profile_passes_enabled:
            self.gas_solver.gpu_pipeline.reset_pass_profile()
        with self._profile_pass("gas"):
            self.gas_solver.step(self, dt)
        gas_profile = getattr(getattr(self.gas_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(gas_profile, dict):
            self.last_pass_profile["gas"] = gas_profile
        if self.profile_passes_enabled:
            self.heat_solver.gpu_pipeline.reset_pass_profile()
        with self._profile_pass("heat"):
            self.heat_solver.step(self, dt)
        heat_profile = getattr(getattr(self.heat_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(heat_profile, dict):
            self.last_pass_profile["heat"] = heat_profile
        self.reaction_solver.reset_runtime_state(self)
        if self.profile_passes_enabled:
            self.reaction_solver.gpu_pipeline.reset_pass_profile()
        with self._profile_pass("reactions before motion"):
            self.reaction_solver.gpu_pipeline.begin_formal_reaction_segment(self, "before_motion")
            try:
                with self._profile_pass("reaction_timed"):
                    self.reaction_solver._advance_timed_slots(self)
                with self._profile_pass("reaction_self"):
                    self.reaction_solver._run_self_rules(self)
                with self._profile_pass("reaction_material_material"):
                    self.reaction_solver._run_material_material(self)
                with self._profile_pass("reaction_material_gas"):
                    self.reaction_solver._run_material_gas(self)
                with self._profile_pass("reaction_material_light"):
                    self.reaction_solver._run_material_light(self)
                with self._profile_pass("reaction_gas_gas"):
                    self.reaction_solver._run_gas_gas(self)
                with self._profile_pass("reaction_gas_light"):
                    self.reaction_solver._run_gas_light(self)
                self.reaction_solver.gpu_pipeline.flush_formal_reaction_segment(self, "before_motion")
            finally:
                self.reaction_solver.gpu_pipeline.end_formal_reaction_segment(self, "before_motion")
        with self._profile_pass("liquid_pre_motion_intent"):
            self.liquid_solver.prepare_motion_flow_intent(self)
        with self._profile_pass("motion"):
            self.motion_solver.step(self, dt)
        motion_profile = getattr(getattr(self.motion_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(motion_profile, dict):
            self.last_pass_profile["motion"] = motion_profile
        with self._profile_pass("liquid"):
            self.liquid_solver.step(self)
        liquid_profile = getattr(getattr(self.liquid_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(liquid_profile, dict):
            self.last_pass_profile["liquid"] = liquid_profile
        with self._profile_pass("optics"):
            self.optics_solver.step(self)
        optics_profile = getattr(self.optics_solver, "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(optics_profile, dict):
            self.last_pass_profile["optics"] = optics_profile
        with self._profile_pass("latch_clear"):
            if self.reaction_solver.gpu_pipeline.clear_reaction_latches(self):
                self.reaction_solver._note_runtime_backend("gpu")
            else:
                self._require_gpu_stage("reaction latch clearing")
                self.cell_flags &= np.uint8(~int(CellFlag.REACTION_LATCHED) & 0xFF)
                self.reaction_solver._note_runtime_backend("cpu")
        reaction_profile = getattr(getattr(self.reaction_solver, "gpu_pipeline", None), "last_pass_profile", None)
        if self.profile_passes_enabled and isinstance(reaction_profile, dict):
            self.last_pass_profile["reactions"] = reaction_profile
        with self._profile_pass("active_decay"):
            active_scheduler_gpu_authoritative = (
                self.simulation_backend == "gpu"
                and "active_tile_ttl" in self.bridge.gpu_authoritative_resources
            )
            if active_scheduler_gpu_authoritative:
                if not self.bridge.decay_active_scheduler(self):
                    self._require_gpu_stage("active scheduler decay")
                    raise RuntimeError("GPU active scheduler decay failed; CPU fallback is disabled")
            elif self.simulation_backend == "gpu":
                self._require_gpu_stage("active scheduler decay")
            else:
                self.active.decay()
        bridge_world_synced = False
        if capture_output:
            if self.simulation_backend != "gpu":
                self.bridge.sync_world(self)
                bridge_world_synced = True
            else:
                self.bridge.sync_force_sources(self)
            bridge_upload_snapshot = self.serialize_bridge_upload_snapshot()
            bridge_frame_snapshot = self.serialize_bridge_frame_snapshot()
        with self._profile_pass("readback"):
            self._finish_readbacks(world_synced=bridge_world_synced)
            self._collect_ready_readbacks(self.frame_id)
        self._bridge_inputs_prepared = False

        if not capture_output:
            return None
        return WorldFrameOutput(
            frame_id=self.frame_id,
            submission_id=frame_input.submission_id if frame_input is not None else None,
            controller_state=output_controller_state,
            consumed_readbacks=consumed_readbacks,
            resolved_targets=resolved_targets,
            resolved_change_intents=resolved_change_intents,
            resolved_carrier_intents=resolved_carrier_intents,
            observations=observations,
            entity_feedback=entity_feedback,
            paging_updates=paging_updates,
            observation_plans=observation_plans,
            readback_plans=readback_plans,
            bridge_upload_snapshot=bridge_upload_snapshot,
            bridge_frame_snapshot=bridge_frame_snapshot,
            queued_observations=queued_observations,
            queued_readbacks=queued_readbacks,
            queued_commands=queued_commands,
            placeholder_count=placeholder_count,
        )

    def _queue_persistent_entity_observations(self) -> list[dict[str, Any]]:
        if not self.entity_states:
            return []
        _, observation_targets = self._frame_entities_to_placeholders_and_observations(list(self.entity_states.values()))
        observation_pairs = self._build_observation_request_pairs(observation_targets, {})
        observation_pairs = [
            (target, self._assign_readback_request_id(request))
            for target, request in observation_pairs
        ]
        observation_requests = [request for _, request in observation_pairs]
        self.pending_readbacks.extend(observation_requests)
        self.bridge_frame_readback_requests.extend(replace(request) for request in observation_requests)
        return [
            self._serialize_observation_plan_for_target_request(target, request)
            for target, request in observation_pairs
        ]

    def _apply_frame_input(
        self,
        frame_input: WorldFrameInput,
    ) -> tuple[
        Any,
        list[PageStripeUpdate],
        dict[str, ResolvedTarget],
        dict[str, ResolvedChangeIntent],
        dict[str, ResolvedCarrierIntent],
        list[dict[str, Any]],
        list[dict[str, Any]],
        int,
        int,
        int,
        int,
    ]:
        paging_updates: list[PageStripeUpdate] = []
        resolved_targets: dict[str, ResolvedTarget] = {}
        resolved_change_intents: dict[str, ResolvedChangeIntent] = {}
        resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = {}
        observation_plans: list[dict[str, Any]] = []
        readback_plans: list[dict[str, Any]] = []
        queued_observations = 0
        queued_readbacks = 0
        queued_commands = 0
        placeholder_count = 0
        if frame_input.controller_state_provided:
            self.controller_state_snapshot = deepcopy(frame_input.controller_state)
        output_controller_state = deepcopy(self.controller_state_snapshot)
        if frame_input.focus_center is not None:
            paging_updates = self.advance_paging(
                frame_input.focus_center[0],
                frame_input.focus_center[1],
                immediate=True,
            )
        placeholder_inputs = [
            self._frame_entity_placeholder_input(placeholder)
            for placeholder in (frame_input.entity_placeholders or [])
        ]
        if frame_input.entities is not None:
            entity_placeholders, _ = self._sync_entity_states(
                [self._frame_entity_state_input(entity) for entity in frame_input.entities]
            )
            placeholder_inputs = entity_placeholders + placeholder_inputs
        if frame_input.entity_placeholders is not None or frame_input.entities is not None:
            self._sync_entity_placeholders(placeholder_inputs)
            placeholder_count = len(placeholder_inputs)
        if frame_input.force_sources is not None:
            self._sync_force_sources(
                [self._frame_force_source_input(force_source) for force_source in frame_input.force_sources]
            )
        if frame_input.emitters is not None:
            self._sync_persistent_emitters(
                [self._frame_emitter_input(emitter) for emitter in frame_input.emitters]
            )
        resolved_targets = self._resolve_target_queries(frame_input.target_queries)
        resolved_change_intents, generated_commands = self._resolve_change_intents(frame_input.change_intents, resolved_targets)
        resolved_carrier_intents, generated_carrier_commands = self._resolve_carrier_intents(
            frame_input.carrier_intents,
            resolved_targets,
        )
        observation_pairs = self._build_observation_request_pairs(frame_input.observation_targets, resolved_targets)
        observation_pairs = [
            (target, self._assign_readback_request_id(request))
            for target, request in observation_pairs
        ]
        observation_requests = [request for _, request in observation_pairs]
        self.pending_readbacks.extend(observation_requests)
        self.bridge_frame_readback_requests.extend(replace(request) for request in observation_requests)
        observation_plans = [
            self._serialize_observation_plan_for_target_request(target, request)
            for target, request in observation_pairs
        ]
        queued_observations = len(observation_requests)
        for command in (
            generated_commands
            + generated_carrier_commands
            + self._resolve_targeted_commands(frame_input.commands, resolved_targets)
        ):
            self.queue_command(command.kind, **command.payload)
            queued_commands += 1
        readback_requests = self._resolve_readback_requests(frame_input.readback_requests, resolved_targets)
        readback_requests = [self._assign_readback_request_id(request) for request in readback_requests]
        self.pending_readbacks.extend(readback_requests)
        self.bridge_frame_readback_requests.extend(replace(request) for request in readback_requests)
        readback_plans = self._serialize_readback_plans_for_requests(readback_requests)
        queued_readbacks = len(readback_requests)
        public_resolved_targets = {
            query_id: self._public_resolved_target(target)
            for query_id, target in resolved_targets.items()
        }
        public_resolved_change_intents = {
            intent_id: self._public_resolved_change_intent(intent)
            for intent_id, intent in resolved_change_intents.items()
        }
        public_resolved_carrier_intents = {
            intent_id: self._public_resolved_carrier_intent(intent)
            for intent_id, intent in resolved_carrier_intents.items()
        }
        return (
            output_controller_state,
            paging_updates,
            public_resolved_targets,
            public_resolved_change_intents,
            public_resolved_carrier_intents,
            observation_plans,
            readback_plans,
            queued_observations,
            queued_readbacks,
            queued_commands,
            placeholder_count,
        )

    def _prepare_preview_frame_context(
        self,
        frame_input: WorldFrameInput,
    ) -> tuple[
        list[PageStripeUpdate],
        list[tuple[PageStripeUpdate, dict[str, Any]]],
        list[ObservationTarget],
        list[EntityPlaceholder],
        int,
    ]:
        paging = deepcopy(self.paging)
        paging_updates: list[PageStripeUpdate] = []
        preview_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
        if frame_input.focus_center is not None:
            paging_updates = paging.focus_on(frame_input.focus_center[0], frame_input.focus_center[1])
        self.paging = paging
        if paging_updates:
            preview_page_stripes = self._preview_apply_paging_updates(paging_updates)

        entity_observation_targets: list[ObservationTarget] = []
        if frame_input.entities is None:
            preview_entity_states = dict(self.entity_states)
            derived_placeholders: list[EntityPlaceholder] = []
            _, entity_observation_targets = self._frame_entities_to_placeholders_and_observations(
                list(preview_entity_states.values())
            )
        else:
            frame_entities = [self._frame_entity_state_input(entity) for entity in frame_input.entities]
            preview_entity_states = {entity.entity_id: entity for entity in frame_entities}
            derived_placeholders, entity_observation_targets = self._frame_entities_to_placeholders_and_observations(frame_entities)
        self.entity_states = preview_entity_states

        placeholder_inputs = [
            self._frame_entity_placeholder_input(placeholder)
            for placeholder in (frame_input.entity_placeholders or [])
        ]
        placeholder_count = 0
        if frame_input.entities is not None:
            placeholder_inputs = derived_placeholders + placeholder_inputs
        if frame_input.entities is not None or frame_input.entity_placeholders is not None:
            preview_placeholders, blocked_cells, released_cells = self._build_preview_entity_placeholders(placeholder_inputs)
            self.entity_placeholders = preview_placeholders
            self._resolver_blocked_cells = blocked_cells
            self._resolver_released_cells = released_cells
            placeholder_count = len(placeholder_inputs)
        else:
            self._resolver_blocked_cells = None
            self._resolver_released_cells = None
        return (
            paging_updates,
            preview_page_stripes,
            entity_observation_targets,
            placeholder_inputs,
            placeholder_count,
        )

    def _snapshot_preview_runtime_state(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id.copy(),
            "phase": self.phase.copy(),
            "cell_flags": self.cell_flags.copy(),
            "velocity": self.velocity.copy(),
            "cell_temperature": self.cell_temperature.copy(),
            "timer_pack": self.timer_pack.copy(),
            "integrity": self.integrity.copy(),
            "island_id": self.island_id.copy(),
            "entity_id": self.entity_id.copy(),
            "placeholder_displaced_material": self.placeholder_displaced_material.copy(),
            "collapse_delay_pending": self.collapse_delay_pending.copy(),
            "flow_velocity": self.flow_velocity.copy(),
            "ambient_temperature": self.ambient_temperature.copy(),
            "pressure_ping": self.pressure_ping.copy(),
            "gas_concentration": self.gas_concentration.copy(),
            "visible_illumination": self.visible_illumination.copy(),
            "cell_optical_dose": self.cell_optical_dose.copy(),
            "gas_optical_dose": self.gas_optical_dose.copy(),
            "active": deepcopy(self.active),
            "islands": deepcopy(self.islands),
            "next_island_id": int(self.next_island_id),
            "collapse_dirty_regions": list(self.collapse_dirty_regions),
            "collapse_deferred_regions": list(self.collapse_deferred_regions),
        }

    def _restore_preview_runtime_state(self, snapshot: dict[str, Any]) -> None:
        self.material_id = snapshot["material_id"]
        self.phase = snapshot["phase"]
        self.cell_flags = snapshot["cell_flags"]
        self.velocity = snapshot["velocity"]
        self.cell_temperature = snapshot["cell_temperature"]
        self.timer_pack = snapshot["timer_pack"]
        self.integrity = snapshot["integrity"]
        self.island_id = snapshot["island_id"]
        self.entity_id = snapshot["entity_id"]
        self.placeholder_displaced_material = snapshot["placeholder_displaced_material"]
        self.collapse_delay_pending = snapshot["collapse_delay_pending"]
        self.flow_velocity = snapshot["flow_velocity"]
        self.ambient_temperature = snapshot["ambient_temperature"]
        self.pressure_ping = snapshot["pressure_ping"]
        self.gas_concentration = snapshot["gas_concentration"]
        self.visible_illumination = snapshot["visible_illumination"]
        self.cell_optical_dose = snapshot["cell_optical_dose"]
        self.gas_optical_dose = snapshot["gas_optical_dose"]
        self.active = snapshot["active"]
        self.islands = deepcopy(snapshot["islands"])
        self.next_island_id = int(snapshot["next_island_id"])
        self.collapse_dirty_regions = snapshot["collapse_dirty_regions"]
        self.collapse_deferred_regions = snapshot["collapse_deferred_regions"]

    def _contextualize_page_stripe_update(self, update: PageStripeUpdate) -> PageStripeUpdate:
        if update.axis == "x":
            default_cross_start = int(self.paging.origin_y)
            default_cross_end = int(self.paging.origin_y + self.height)
        else:
            default_cross_start = int(self.paging.origin_x)
            default_cross_end = int(self.paging.origin_x + self.width)
        cross_world_start = default_cross_start if update.cross_world_start is None else int(update.cross_world_start)
        cross_world_end = default_cross_end if update.cross_world_end is None else int(update.cross_world_end)
        if (
            update.cross_world_start == cross_world_start
            and update.cross_world_end == cross_world_end
        ):
            return update
        return replace(
            update,
            cross_world_start=cross_world_start,
            cross_world_end=cross_world_end,
        )

    @staticmethod
    def _page_store_key(update: PageStripeUpdate) -> tuple[str, int, int, int, int]:
        return (
            str(update.axis),
            int(update.world_start),
            int(update.world_end),
            0 if update.cross_world_start is None else int(update.cross_world_start),
            0 if update.cross_world_end is None else int(update.cross_world_end),
        )

    def _preview_apply_paging_updates(
        self,
        updates: list[PageStripeUpdate],
    ) -> list[tuple[PageStripeUpdate, dict[str, Any]]]:
        preview_saved_payloads: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
        preview_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
        for update in updates:
            if update.kind != "save":
                continue
            preview_saved_payloads[self._page_store_key(update)] = self.capture_page_stripe(update)
            self._clear_saved_page_stripe_runtime_state(update)
        for update in updates:
            if update.kind != "load":
                continue
            payload = preview_saved_payloads.get(self._page_store_key(update))
            if payload is None:
                payload = self.page_store.load(update)
            if payload is None:
                payload = self._default_page_stripe_payload(update)
            self._apply_page_stripe(update, payload)
            preview_page_stripes.append((PageStripeUpdate(**asdict(update)), deepcopy(payload)))
        return preview_page_stripes

    def _preview_bridge_placeholder_dirty_rects(
        self,
        current_entity_placeholders: dict[int, set[tuple[int, int]]],
        placeholders: list[EntityPlaceholder],
    ) -> list[dict[str, Any]]:
        current_cells = {
            cell: entity_id
            for entity_id, cells in current_entity_placeholders.items()
            for cell in cells
        }
        next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
        for placeholder in placeholders:
            for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
                for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                    if not self.in_bounds(x, y):
                        continue
                    next_cells[(x, y)] = placeholder

        changed_cells: set[tuple[int, int]] = set()
        for cell, entity_id in current_cells.items():
            next_placeholder = next_cells.get(cell)
            if next_placeholder is None or next_placeholder.entity_id != entity_id:
                changed_cells.add(cell)
        for cell, placeholder in next_cells.items():
            x, y = cell
            material_id = int(self.material_id[y, x])
            entity_id = int(self.entity_id[y, x])
            has_matching_placeholder_cell = (
                material_id > 0
                and self._shadow_material_is_placeholder(material_id)
                and entity_id == int(placeholder.entity_id)
            )
            if current_cells.get(cell) != placeholder.entity_id or not has_matching_placeholder_cell:
                changed_cells.add(cell)

        payload: list[dict[str, Any]] = []
        for x, y in sorted(changed_cells):
            world_rect = self._buffer_bbox_to_world_bbox((int(x), int(y), int(x) + 1, int(y) + 1))
            payload.append(
                {
                    "buffer_rect": [int(x), int(y), int(x) + 1, int(y) + 1],
                    "world_rect": [int(world_rect[0]), int(world_rect[1]), int(world_rect[2]), int(world_rect[3])],
                }
            )
        return payload

    @staticmethod
    def _serialize_bridge_readback_request_stages(
        requests: list[ReadbackRequest],
        *,
        stage: str | None = None,
        reserved_request_ids: set[int] | None = None,
        observation_request_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        reserved_ids = set() if reserved_request_ids is None else {int(request_id) for request_id in reserved_request_ids}
        observation_ids = set() if observation_request_ids is None else {int(request_id) for request_id in observation_request_ids}
        payload: list[dict[str, Any]] = []
        for request in requests:
            if request.request_id is None:
                continue
            if stage is None:
                current_stage = "predicted"
                if int(request.request_id) in reserved_ids:
                    current_stage = "reserved"
                elif int(request.request_id) in observation_ids:
                    current_stage = "predicted"
            else:
                current_stage = str(stage)
            payload.append({"request_id": int(request.request_id), "stage": current_stage})
        return payload

    @staticmethod
    def _serialize_bridge_index_stages(
        values: Sequence[Any],
        *,
        stage: str,
    ) -> list[dict[str, Any]]:
        return [{"index": int(index), "stage": str(stage)} for index, _ in enumerate(values)]

    def _serialize_preview_bridge_frame_snapshot(
        self,
        *,
        current_entity_placeholders: dict[int, set[tuple[int, int]]],
        resolved_commands: list[WorldCommand],
        observation_requests: list[ReadbackRequest],
        readback_requests: list[ReadbackRequest],
        placeholder_inputs: list[EntityPlaceholder],
        paging_updates: list[PageStripeUpdate],
        page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]],
        reserved_readback_request_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        bridge_input_stage = "reserved" if reserved_readback_request_ids is not None else "predicted"
        snapshot_prepared = bool(
            resolved_commands
            or observation_requests
            or readback_requests
            or placeholder_inputs
            or paging_updates
            or page_stripes
        )
        serialized_page_stripes = [
            {
                "update": self.serialize_page_stripe_update(update),
                "payload": self.serialize_page_stripe_payload(payload),
            }
            for update, payload in page_stripes
        ]
        return {
            "prepared": snapshot_prepared,
            "commands": [self.serialize_world_command(command) for command in resolved_commands],
            "command_stages": self._serialize_bridge_index_stages(
                resolved_commands,
                stage=bridge_input_stage,
            ),
            "readback_requests": [
                self.serialize_readback_request(request)
                for request in [*observation_requests, *readback_requests]
            ],
            "readback_request_stages": self._serialize_bridge_readback_request_stages(
                [*observation_requests, *readback_requests],
                reserved_request_ids=reserved_readback_request_ids,
                observation_request_ids={
                    int(request.request_id)
                    for request in observation_requests
                    if request.request_id is not None
                },
            ),
            "placeholders": [
                self.serialize_entity_placeholder_input(placeholder) for placeholder in placeholder_inputs
            ],
            "placeholder_stages": self._serialize_bridge_index_stages(
                placeholder_inputs,
                stage=bridge_input_stage,
            ),
            "placeholder_dirty_rects": self._preview_bridge_placeholder_dirty_rects(
                current_entity_placeholders,
                placeholder_inputs,
            ),
            "paging_updates": [
                self.serialize_page_stripe_update(update) for update in paging_updates
            ],
            "paging_update_stages": self._serialize_bridge_index_stages(
                paging_updates,
                stage=bridge_input_stage,
            ),
            "page_stripes": serialized_page_stripes,
            "page_stripe_stages": self._serialize_bridge_index_stages(
                page_stripes,
                stage=bridge_input_stage,
            ),
        }

    def _queue_loaded_collapse_pending_regions(self, update: PageStripeUpdate) -> None:
        if update.kind != "load":
            return
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                pending = self.collapse_delay_pending[:, start:end]
                ys, xs = np.nonzero(pending)
                if ys.size == 0:
                    continue
                self.collapse_deferred_regions.append(
                    (
                        max(0, start + int(xs.min()) - 1),
                        max(0, int(ys.min()) - 1),
                        min(self.width, start + int(xs.max()) + 2),
                        min(self.height, int(ys.max()) + 2),
                    )
                )
                continue
            pending = self.collapse_delay_pending[start:end, :]
            ys, xs = np.nonzero(pending)
            if ys.size == 0:
                continue
            self.collapse_deferred_regions.append(
                (
                    max(0, int(xs.min()) - 1),
                    max(0, start + int(ys.min()) - 1),
                    min(self.width, int(xs.max()) + 2),
                    min(self.height, start + int(ys.max()) + 2),
                )
            )

    def _clear_saved_page_stripe_runtime_state(self, update: PageStripeUpdate) -> None:
        if update.kind != "save":
            return
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                self.active.clear_rect(start, 0, end, self.height)
            else:
                self.active.clear_rect(0, start, self.width, end)
        self.collapse_dirty_regions = self._prune_page_stripe_regions(self.collapse_dirty_regions, update)
        self.collapse_deferred_regions = self._prune_page_stripe_regions(self.collapse_deferred_regions, update)

    def _prune_page_stripe_regions(
        self,
        regions: Iterable[tuple[int, int, int, int]],
        update: PageStripeUpdate,
    ) -> list[tuple[int, int, int, int]]:
        next_regions = [tuple(int(value) for value in region) for region in regions]
        if update.kind != "save":
            return next_regions
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            pruned: list[tuple[int, int, int, int]] = []
            for region in next_regions:
                pruned.extend(self._subtract_page_stripe_range_from_region(region, axis=update.axis, start=start, end=end))
            next_regions = pruned
        return next_regions

    @staticmethod
    def _subtract_page_stripe_range_from_region(
        region: tuple[int, int, int, int],
        *,
        axis: str,
        start: int,
        end: int,
    ) -> list[tuple[int, int, int, int]]:
        x0, y0, x1, y1 = (int(value) for value in region)
        if x1 <= x0 or y1 <= y0:
            return []
        if axis == "x":
            overlap0 = max(x0, int(start))
            overlap1 = min(x1, int(end))
            if overlap0 >= overlap1:
                return [(x0, y0, x1, y1)]
            remaining: list[tuple[int, int, int, int]] = []
            if x0 < overlap0:
                remaining.append((x0, y0, overlap0, y1))
            if overlap1 < x1:
                remaining.append((overlap1, y0, x1, y1))
            return remaining
        overlap0 = max(y0, int(start))
        overlap1 = min(y1, int(end))
        if overlap0 >= overlap1:
            return [(x0, y0, x1, y1)]
        remaining = []
        if y0 < overlap0:
            remaining.append((x0, y0, x1, overlap0))
        if overlap1 < y1:
            remaining.append((x0, overlap1, x1, y1))
        return remaining

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.gas_solver.release()
        self.heat_solver.release()
        self.collapse_solver.release()
        self.motion_solver.release()
        self.liquid_solver.release()
        self.optics_solver.release()
        self.reaction_solver.release()
        self.placeholder_pipeline.release()
        self.page_stripe_pipeline.release()
        self.grid_command_pipeline.release()
        self.bridge.release()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def material_by_id(self, material_id: int) -> MaterialDef:
        material = self._shadow_material_def(int(material_id))
        if material is None:
            raise KeyError(material_id)
        return material

    def allocate_island_id(self) -> int:
        island_id = max(1, int(self.next_island_id))
        while island_id in self.islands:
            island_id += 1
        self.next_island_id = island_id + 1
        return island_id

    def _refresh_island_records_for_ids(self, island_ids: Iterable[int]) -> None:
        touched = {int(island_id) for island_id in island_ids if int(island_id) > 0}
        if not touched:
            return
        for island_id in touched:
            invalid_mask = (self.island_id == island_id) & (
                (self.phase != int(Phase.FALLING_ISLAND)) | (self.material_id <= 0)
            )
            if np.any(invalid_mask):
                self.island_id[invalid_mask] = 0
            coords = np.argwhere(
                (self.island_id == island_id)
                & (self.phase == int(Phase.FALLING_ISLAND))
                & (self.material_id > 0)
            )
            if coords.size == 0:
                self.islands.pop(island_id, None)
                continue
            min_y, min_x = coords.min(axis=0).tolist()
            max_y, max_x = coords.max(axis=0).tolist()
            previous = self.islands.get(island_id)
            if previous is None:
                velocity_xy = tuple(np.mean(self.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32).tolist())
                subcell_offset = (0.0, 0.0)
            else:
                velocity_xy = (float(previous.velocity_xy[0]), float(previous.velocity_xy[1]))
                subcell_offset = (float(previous.subcell_offset[0]), float(previous.subcell_offset[1]))
            self.islands[island_id] = FallingIslandRecord(
                island_id=island_id,
                bbox=(int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1),
                velocity_xy=(float(velocity_xy[0]), float(velocity_xy[1])),
                subcell_offset=subcell_offset,
            )
        self.next_island_id = max(int(self.next_island_id), max(self.islands, default=0) + 1, 1)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def cell_xy_to_gas(self, x: int, y: int) -> tuple[int, int]:
        """Map a cell-space (x, y) pair onto the lower-resolution gas grid."""
        return (
            min(self.gas_height - 1, max(0, y // self.gas_cell_size)),
            min(self.gas_width - 1, max(0, x // self.gas_cell_size)),
        )

    def cell_to_gas(self, y: int, x: int) -> tuple[int, int]:
        """Map a cell-space (y, x) pair onto the lower-resolution gas grid."""
        return self.cell_xy_to_gas(x, y)

    def sample_ambient_to_cells(self) -> np.ndarray:
        return np.repeat(np.repeat(self.ambient_temperature, self.gas_cell_size, axis=0), self.gas_cell_size, axis=1)[: self.height, : self.width]

    def ambient_temperature_at_cell(self, x: int, y: int) -> float:
        gy, gx = self.cell_xy_to_gas(x, y)
        return float(self.ambient_temperature[gy, gx])

    def ambient_temperature_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        if x0 >= x1 or y0 >= y1:
            return np.zeros((0, 0), dtype=np.float32)
        gx0 = min(self.gas_width, max(0, x0 // self.gas_cell_size))
        gy0 = min(self.gas_height, max(0, y0 // self.gas_cell_size))
        gx1 = min(self.gas_width, max(gx0 + 1, (x1 + self.gas_cell_size - 1) // self.gas_cell_size))
        gy1 = min(self.gas_height, max(gy0 + 1, (y1 + self.gas_cell_size - 1) // self.gas_cell_size))
        repeated = np.repeat(
            np.repeat(self.ambient_temperature[gy0:gy1, gx0:gx1], self.gas_cell_size, axis=0),
            self.gas_cell_size,
            axis=1,
        )
        local_y0 = y0 - gy0 * self.gas_cell_size
        local_x0 = x0 - gx0 * self.gas_cell_size
        return np.ascontiguousarray(repeated[local_y0 : local_y0 + (y1 - y0), local_x0 : local_x0 + (x1 - x0)])

    def sample_flow_to_cells(self) -> np.ndarray:
        return np.repeat(np.repeat(self.flow_velocity, self.gas_cell_size, axis=0), self.gas_cell_size, axis=1)[: self.height, : self.width]

    def downsample_cells_to_gas(self, field: np.ndarray) -> np.ndarray:
        result = np.zeros((self.gas_height, self.gas_width), dtype=np.float32)
        for gy in range(self.gas_height):
            for gx in range(self.gas_width):
                x0 = gx * self.gas_cell_size
                y0 = gy * self.gas_cell_size
                block = field[y0 : min(self.height, y0 + self.gas_cell_size), x0 : min(self.width, x0 + self.gas_cell_size)]
                result[gy, gx] = float(block.mean()) if block.size else 0.0
        return result

    def add_gas_from_cells(self, mask: np.ndarray, species: str, amount: float) -> None:
        species_id = self._resolve_sanctioned_gas_id(species)
        if species_id < 0:
            raise KeyError(species)
        self._invalidate_gpu_authoritative_resources("gas_concentration")
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            gy, gx = self.cell_to_gas(y, x)
            self.gas_concentration[species_id, gy, gx] += amount

    def set_cell_by_id(self, x: int, y: int, material_id: int, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
        self._invalidate_gpu_authoritative_cell_resources()
        previous_material = int(self.material_id[y, x])
        previous_phase = int(self.phase[y, x])
        previous_island_id = int(self.island_id[y, x])
        previous_displaced = int(self.placeholder_displaced_material[y, x])
        self.material_id[y, x] = int(material_id)
        if phase is not None:
            resolved_phase = int(phase)
        else:
            shadow_phase = self._shadow_material_default_phase(int(material_id))
            resolved_phase = int(shadow_phase) if shadow_phase is not None else 0
        self.phase[y, x] = resolved_phase
        self.cell_flags[y, x] = 0
        self.timer_pack[y, x] = 0
        shadow_integrity = self._shadow_material_base_integrity(int(material_id))
        self.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
        spawn_temperature = self._shadow_material_spawn_temperature(int(material_id))
        if spawn_temperature is not None:
            self.cell_temperature[y, x] = max(float(self.cell_temperature[y, x]), spawn_temperature)
        current_is_placeholder = self._shadow_material_is_placeholder(int(material_id))
        previous_is_placeholder = self._shadow_material_is_placeholder(previous_material)
        if current_is_placeholder:
            if previous_is_placeholder:
                self.placeholder_displaced_material[y, x] = previous_displaced
            elif previous_material != 0 and previous_phase == int(Phase.LIQUID):
                self.placeholder_displaced_material[y, x] = previous_material
            else:
                self.placeholder_displaced_material[y, x] = 0
        else:
            self.entity_id[y, x] = 0
            self.placeholder_displaced_material[y, x] = 0
        current_displaced = int(self.placeholder_displaced_material[y, x])
        if current_is_placeholder or previous_is_placeholder or previous_displaced != current_displaced:
            self._pending_placeholder_dirty_rects.append((int(x), int(y), int(x) + 1, int(y) + 1))
        self.island_id[y, x] = 0 if self.phase[y, x] != int(Phase.FALLING_ISLAND) else self.island_id[y, x]
        self._refresh_island_records_for_ids((previous_island_id, int(self.island_id[y, x])))
        self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
        if mark_dirty and (
            self._cell_participates_in_collapse(previous_material, previous_phase)
            or self._cell_participates_in_collapse(int(self.material_id[y, x]), int(self.phase[y, x]))
        ):
            self._mark_collapse_dirty_rect(x, y, x + 1, y + 1)

    def _inject_velocity_immediate(
        self,
        x: int,
        y: int,
        velocity: tuple[float, float],
        radius: int,
        *,
        carrier: str,
        mode: str,
    ) -> None:
        velocity_vec = np.asarray(velocity, dtype=np.float32)
        if mode not in {"add", "set"}:
            raise ValueError(f"unsupported velocity mode: {mode}")
        if carrier not in {"cell", "flow", "both"}:
            raise ValueError(f"unsupported velocity carrier: {carrier}")
        if carrier in {"cell", "both"}:
            self._invalidate_gpu_authoritative_resources("cell_core")
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            cell_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            if mode == "set":
                self.velocity[cell_mask] = velocity_vec
            else:
                self.velocity[cell_mask] += velocity_vec
        if carrier in {"flow", "both"}:
            self._invalidate_gpu_authoritative_resources("flow_velocity")
            gas_center_x = min(self.gas_width - 1, max(0, x // self.gas_cell_size))
            gas_center_y = min(self.gas_height - 1, max(0, y // self.gas_cell_size))
            gas_radius = max(0, (radius + self.gas_cell_size - 1) // self.gas_cell_size)
            yy, xx = np.mgrid[0:self.gas_height, 0:self.gas_width]
            flow_mask = (xx - gas_center_x) ** 2 + (yy - gas_center_y) ** 2 <= gas_radius ** 2
            if mode == "set":
                self.flow_velocity[flow_mask] = velocity_vec
            else:
                self.flow_velocity[flow_mask] += velocity_vec
        self._mark_active_rect_runtime(
            max(0, x - radius),
            max(0, y - radius),
            min(self.width, x + radius + 1),
            min(self.height, y + radius + 1),
        )

    def _inject_temperature_immediate(self, x: int, y: int, delta: float, radius: int) -> None:
        self._invalidate_gpu_authoritative_resources("cell_core")
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        self.cell_temperature[mask] += delta
        self._mark_active_rect_runtime(max(0, x - radius), max(0, y - radius), min(self.width, x + radius + 1), min(self.height, y + radius + 1))

    def _inject_gas_immediate(self, x: int, y: int, species: str, amount: float, radius: int) -> None:
        gy, gx = self.cell_xy_to_gas(x, y)
        species_id = self._resolve_sanctioned_gas_id(species)
        if species_id < 0:
            raise KeyError(species)
        self._invalidate_gpu_authoritative_resources("gas_concentration")
        self.gas_concentration[species_id, gy, gx] = max(0.0, self.gas_concentration[species_id, gy, gx] + amount)
        self._mark_active_rect_runtime(max(0, x - radius), max(0, y - radius), min(self.width, x + radius + 1), min(self.height, y + radius + 1))

    def set_cell(self, x: int, y: int, material_name: str, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
        material_id = self._resolve_sanctioned_material_id(material_name)
        if material_id <= 0:
            raise KeyError(material_name)
        self.set_cell_by_id(x, y, material_id, phase=phase, mark_dirty=mark_dirty)

    def clear_cell(self, x: int, y: int, *, mark_dirty: bool = True) -> None:
        self._invalidate_gpu_authoritative_cell_resources()
        previous_material = int(self.material_id[y, x])
        previous_phase = int(self.phase[y, x])
        previous_island_id = int(self.island_id[y, x])
        previous_displaced = int(self.placeholder_displaced_material[y, x])
        previous_is_placeholder = self._shadow_material_is_placeholder(previous_material)
        self.material_id[y, x] = 0
        self.phase[y, x] = 0
        self.cell_flags[y, x] = 0
        self.velocity[y, x] = 0.0
        self.cell_temperature[y, x] = self.ambient_temperature_at_cell(x, y)
        self.timer_pack[y, x] = 0
        self.integrity[y, x] = 0.0
        self.island_id[y, x] = 0
        self.entity_id[y, x] = 0
        self.placeholder_displaced_material[y, x] = 0
        if previous_is_placeholder or previous_displaced != 0:
            self._pending_placeholder_dirty_rects.append((int(x), int(y), int(x) + 1, int(y) + 1))
        self._refresh_island_records_for_ids((previous_island_id,))
        self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
        if mark_dirty and self._cell_participates_in_collapse(previous_material, previous_phase):
            self._mark_collapse_dirty_rect(x, y, x + 1, y + 1)

    def clear_cells(self, mask: np.ndarray, *, mark_dirty: bool = True) -> None:
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            self.clear_cell(int(x), int(y), mark_dirty=mark_dirty)

    def set_material_by_mask(self, mask: np.ndarray, material_name: str, *, phase: Phase | None = None, mark_dirty: bool = True) -> None:
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            self.set_cell(int(x), int(y), material_name, phase=phase, mark_dirty=mark_dirty)

    def swap_cells(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._invalidate_gpu_authoritative_cell_resources()
        previous_placeholder_cells = (
            (x0, y0, int(self.entity_id[y0, x0]), int(self.material_id[y0, x0])),
            (x1, y1, int(self.entity_id[y1, x1]), int(self.material_id[y1, x1])),
        )
        previous_island_ids = (
            int(self.island_id[y0, x0]),
            int(self.island_id[y1, x1]),
        )
        material0 = int(self.material_id[y0, x0])
        material1 = int(self.material_id[y1, x1])
        if (material0 == 0) != (material1 == 0):
            src_x, src_y, dst_x, dst_y = (x1, y1, x0, y0) if material0 == 0 else (x0, y0, x1, y1)
            for array in (
                self.material_id,
                self.phase,
                self.cell_flags,
                self.cell_temperature,
                self.integrity,
                self.island_id,
                self.entity_id,
                self.placeholder_displaced_material,
            ):
                value = array[src_y, src_x].copy() if hasattr(array[src_y, src_x], "copy") else array[src_y, src_x]
                array[dst_y, dst_x] = value
            for array in (self.velocity, self.timer_pack):
                array[dst_y, dst_x] = array[src_y, src_x].copy()
            self.material_id[src_y, src_x] = 0
            self.phase[src_y, src_x] = 0
            self.cell_flags[src_y, src_x] = 0
            self.cell_temperature[src_y, src_x] = 0.0
            self.integrity[src_y, src_x] = 0.0
            self.island_id[src_y, src_x] = 0
            self.entity_id[src_y, src_x] = 0
            self.placeholder_displaced_material[src_y, src_x] = 0
            self.velocity[src_y, src_x] = 0.0
            self.timer_pack[src_y, src_x] = 0
        else:
            for array in (
                self.material_id,
                self.phase,
                self.cell_flags,
                self.cell_temperature,
                self.integrity,
                self.island_id,
                self.entity_id,
                self.placeholder_displaced_material,
            ):
                array[y0, x0], array[y1, x1] = (
                    array[y1, x1].copy() if hasattr(array[y1, x1], "copy") else array[y1, x1],
                    array[y0, x0].copy() if hasattr(array[y0, x0], "copy") else array[y0, x0],
                )
            for array in (self.velocity, self.timer_pack):
                temp = array[y0, x0].copy()
                array[y0, x0] = array[y1, x1]
                array[y1, x1] = temp
        for cell_x, cell_y, entity_id, material_id in previous_placeholder_cells:
            if entity_id > 0:
                cells = self.entity_placeholders.get(entity_id)
                if cells is not None:
                    cells.discard((cell_x, cell_y))
                    if not cells:
                        self.entity_placeholders.pop(entity_id, None)
        for cell_x, cell_y in ((x0, y0), (x1, y1)):
            entity_id = int(self.entity_id[cell_y, cell_x])
            material_id = int(self.material_id[cell_y, cell_x])
            if entity_id > 0 and material_id > 0 and self._shadow_material_is_placeholder(material_id):
                self.entity_placeholders.setdefault(entity_id, set()).add((cell_x, cell_y))
        self._refresh_island_records_for_ids(
            previous_island_ids
            + (
                int(self.island_id[y0, x0]),
                int(self.island_id[y1, x1]),
            )
        )
        self._mark_active_rect_runtime(max(0, min(x0, x1) - 1), max(0, min(y0, y1) - 1), min(self.width, max(x0, x1) + 2), min(self.height, max(y0, y1) + 2))

    def clear_cell_region(self, x0: int, y0: int, x1: int, y1: int, *, mark_dirty: bool = True) -> None:
        self._invalidate_gpu_authoritative_cell_resources()
        region_material = self.material_id[y0:y1, x0:x1]
        region_phase = self.phase[y0:y1, x0:x1]
        region_island_id = self.island_id[y0:y1, x0:x1].copy()
        affects_collapse = bool(
            mark_dirty
            and region_material.size
            and np.any(
                (region_material != 0)
                & (region_phase != int(Phase.FALLING_ISLAND))
                & (self.material_is_structural[region_material] | self.material_is_support_anchor[region_material])
            )
        )
        self.material_id[y0:y1, x0:x1] = 0
        self.phase[y0:y1, x0:x1] = 0
        self.cell_flags[y0:y1, x0:x1] = 0
        self.velocity[y0:y1, x0:x1] = 0.0
        self.cell_temperature[y0:y1, x0:x1] = self.ambient_temperature_region(x0, y0, x1, y1)
        self.timer_pack[y0:y1, x0:x1] = 0
        self.integrity[y0:y1, x0:x1] = 0.0
        self.island_id[y0:y1, x0:x1] = 0
        self.entity_id[y0:y1, x0:x1] = 0
        self.placeholder_displaced_material[y0:y1, x0:x1] = 0
        self._refresh_island_records_for_ids(np.unique(region_island_id))
        self._mark_active_rect_runtime(x0, y0, x1, y1)
        if affects_collapse:
            self._mark_collapse_dirty_rect(x0, y0, x1, y1)

    def serialize_local_cells(self, x: int, y: int, width: int, height: int) -> dict[str, Any]:
        world_x0, world_y0, world_x1, world_y1 = self._clamped_world_window(
            int(x),
            int(y),
            int(width),
            int(height),
        )
        size_x = max(0, world_x1 - world_x0)
        size_y = max(0, world_y1 - world_y0)
        material_id = self._extract_world_window(self.material_id, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        phase = self._extract_world_window(self.phase, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        cell_flags = self._extract_world_window(self.cell_flags, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        velocity = self._extract_world_window(self.velocity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        cell_temperature = self._extract_world_window(
            self.cell_temperature,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        timer_pack = self._extract_world_window(self.timer_pack, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        integrity = self._extract_world_window(self.integrity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        island_id = self._extract_world_window(self.island_id, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        entity_id = self._extract_world_window(self.entity_id, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        placeholder_displaced_material = self._extract_world_window(
            self.placeholder_displaced_material,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        collapse_delay_pending = self._extract_world_window(
            self.collapse_delay_pending,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        return {
            "origin": [world_x0, world_y0],
            "size": [size_x, size_y],
            "material_id": material_id.tolist(),
            "phase": phase.tolist(),
            "cell_flags": cell_flags.tolist(),
            "velocity": velocity.round(4).tolist(),
            "cell_temperature": cell_temperature.round(3).tolist(),
            "temperature": cell_temperature.round(3).tolist(),
            "timer_pack": timer_pack.tolist(),
            "integrity": integrity.round(3).tolist(),
            "island_id": island_id.tolist(),
            "entity_id": entity_id.tolist(),
            "placeholder_displaced_material": placeholder_displaced_material.tolist(),
            "collapse_delay_pending": collapse_delay_pending.astype(np.uint8).tolist(),
        }

    def serialize_temperature_window(self, x: int, y: int, width: int, height: int) -> dict[str, Any]:
        world_x0, world_y0, world_x1, world_y1 = self._clamped_world_window(
            int(x),
            int(y),
            int(width),
            int(height),
        )
        temperature = self._extract_world_window(
            self.cell_temperature,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        return {"temperature": temperature.round(3).tolist()}

    def serialize_gas(self, species: str) -> list[list[float]]:
        species_id = self._resolve_sanctioned_gas_id(species)
        if species_id < 0:
            raise KeyError(species)
        return self._extract_world_window(
            self.gas_concentration[species_id],
            int(self.paging.origin_x) // int(self.gas_cell_size),
            int(self.paging.origin_y) // int(self.gas_cell_size),
            int(self.paging.origin_x) // int(self.gas_cell_size) + int(self.gas_width),
            int(self.paging.origin_y) // int(self.gas_cell_size) + int(self.gas_height),
            x_axis=1,
            y_axis=0,
            gas_grid=True,
        ).round(4).tolist()

    def serialize_pressure(self) -> list[list[float]]:
        return self._extract_world_window(
            self.pressure_ping,
            int(self.paging.origin_x) // int(self.gas_cell_size),
            int(self.paging.origin_y) // int(self.gas_cell_size),
            int(self.paging.origin_x) // int(self.gas_cell_size) + int(self.gas_width),
            int(self.paging.origin_y) // int(self.gas_cell_size) + int(self.gas_height),
            x_axis=1,
            y_axis=0,
            gas_grid=True,
        ).round(4).tolist()

    def serialize_velocity(self) -> list[list[list[float]]]:
        return self._extract_world_window(
            self.flow_velocity,
            int(self.paging.origin_x) // int(self.gas_cell_size),
            int(self.paging.origin_y) // int(self.gas_cell_size),
            int(self.paging.origin_x) // int(self.gas_cell_size) + int(self.gas_width),
            int(self.paging.origin_y) // int(self.gas_cell_size) + int(self.gas_height),
            x_axis=1,
            y_axis=0,
            gas_grid=True,
        ).round(4).tolist()

    def serialize_visible_illumination(self) -> list[list[list[float]]]:
        return self._extract_world_window(
            self.visible_illumination,
            int(self.paging.origin_x),
            int(self.paging.origin_y),
            int(self.paging.origin_x) + int(self.width),
            int(self.paging.origin_y) + int(self.height),
            x_axis=1,
            y_axis=0,
        ).round(4).tolist()

    def serialize_gas_runtime(self) -> dict[str, Any]:
        snapshot = self.gas_solver.runtime_snapshot()
        solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
        species_total = np.asarray(snapshot["species_total_concentration"], dtype=np.float32)
        species_active = np.asarray(snapshot["species_active_concentration"], dtype=np.float32)
        species_runtime = []
        for species_id in range(max(len(self.gas_name_by_id), int(species_total.shape[0]), int(species_active.shape[0]))):
            species_name = self._shadow_gas_name(species_id)
            if not species_name:
                continue
            total = float(species_total[species_id]) if species_id < species_total.shape[0] else 0.0
            active = float(species_active[species_id]) if species_id < species_active.shape[0] else 0.0
            species_runtime.append(
                {
                    "species_id": int(species_id),
                    "species": species_name,
                    "total_concentration": round(total, 4),
                    "active_concentration": round(active, 4),
                }
            )
        return {
            "backend": self.gas_solver.last_backend,
            "pressure_iterations": int(snapshot["pressure_iterations"]),
            "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
            "gas_grid_size": [int(self.gas_width), int(self.gas_height)],
            "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
            "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
            "solve_tile_mask": solve_tile_mask.tolist(),
            "solve_gas_mask": solve_gas_mask.tolist(),
            "force_source_count_before": int(snapshot["force_source_count_before"]),
            "force_source_count_after": int(snapshot["force_source_count_after"]),
            "velocity_changed": bool(snapshot["velocity_changed"]),
            "ambient_changed": bool(snapshot["ambient_changed"]),
            "gas_changed": bool(snapshot["gas_changed"]),
            "pressure_range": np.asarray(snapshot["pressure_range"], dtype=np.float32).round(4).tolist(),
            "ambient_range": np.asarray(snapshot["ambient_range"], dtype=np.float32).round(4).tolist(),
            "flow_speed_range": np.asarray(snapshot["flow_speed_range"], dtype=np.float32).round(4).tolist(),
            "species_runtime": species_runtime,
        }

    def serialize_heat_runtime(self) -> dict[str, Any]:
        snapshot = self.heat_solver.runtime_snapshot()
        solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
        solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
        phase_targets = np.asarray(snapshot["phase_targets"], dtype=np.int32)
        boil_targets = np.asarray(snapshot["boil_targets"], dtype=np.int32)
        condense_targets = np.asarray(snapshot["condense_targets"], dtype=np.bool_)

        phase_payload = []
        public_phase_targets = snapshot.get("public_phase_targets")
        if isinstance(public_phase_targets, list) and public_phase_targets:
            for target in public_phase_targets:
                target_material_id = int(target["target_material_id"])
                phase_payload.append(
                    {
                        "x": int(target["x"]),
                        "y": int(target["y"]),
                        "target_material_id": target_material_id,
                        "target_material": self._shadow_material_name(target_material_id),
                    }
                )
        else:
            phase_ys, phase_xs = np.nonzero(phase_targets > 0)
            for y, x in zip(phase_ys.tolist(), phase_xs.tolist()):
                target_material_id = int(phase_targets[y, x])
                world_x, world_y = self._buffer_to_world_position((int(x), int(y)))
                phase_payload.append(
                    {
                        "x": int(world_x),
                        "y": int(world_y),
                        "target_material_id": target_material_id,
                        "target_material": self._shadow_material_name(target_material_id),
                    }
                )

        boil_payload = []
        public_boil_targets = snapshot.get("public_boil_targets")
        if isinstance(public_boil_targets, list) and public_boil_targets:
            for target in public_boil_targets:
                target_species_id = int(target["target_species_id"])
                boil_payload.append(
                    {
                        "x": int(target["x"]),
                        "y": int(target["y"]),
                        "target_species_id": target_species_id,
                        "target_species": self._shadow_gas_name(target_species_id),
                    }
                )
        else:
            boil_ys, boil_xs = np.nonzero(boil_targets > 0)
            for y, x in zip(boil_ys.tolist(), boil_xs.tolist()):
                target_species_id = int(boil_targets[y, x]) - 1
                world_x, world_y = self._buffer_to_world_position((int(x), int(y)))
                boil_payload.append(
                    {
                        "x": int(world_x),
                        "y": int(world_y),
                        "target_species_id": target_species_id,
                        "target_species": self._shadow_gas_name(target_species_id),
                    }
                )

        condense_payload = []
        public_condense_targets = snapshot.get("public_condense_targets")
        if isinstance(public_condense_targets, list) and public_condense_targets:
            for target in public_condense_targets:
                species_id = int(target["species_id"])
                target_material_id = int(target["target_material_id"])
                condense_payload.append(
                    {
                        "gas_x": int(target["gas_x"]),
                        "gas_y": int(target["gas_y"]),
                        "species_id": species_id,
                        "species": self._shadow_gas_name(species_id),
                        "target_material_id": target_material_id,
                        "target_material": self._shadow_material_name(target_material_id),
                    }
                )
        else:
            species_count = int(condense_targets.shape[0])
            for species_id in range(species_count):
                gas_y, gas_x = np.nonzero(condense_targets[species_id] & (solve_gas_mask > 0))
                if len(gas_y) == 0:
                    continue
                target_material_id = self._shadow_condense_target_material_id(species_id)
                if target_material_id <= 0:
                    continue
                target_material_name = self._shadow_material_name(target_material_id)
                for gy, gx in zip(gas_y.tolist(), gas_x.tolist()):
                    world_gx, world_gy = self._buffer_gas_to_world_position((int(gx), int(gy)))
                    condense_payload.append(
                        {
                            "gas_x": int(world_gx),
                            "gas_y": int(world_gy),
                            "species_id": int(species_id),
                            "species": self._shadow_gas_name(species_id),
                            "target_material_id": target_material_id,
                            "target_material": target_material_name,
                        }
                    )

        return {
            "backend": self.heat_solver.last_backend,
            "ambient_iterations": int(snapshot["ambient_iterations"]),
            "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
            "cell_grid_size": [int(self.width), int(self.height)],
            "gas_grid_size": [int(self.gas_width), int(self.gas_height)],
            "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
            "solve_cell_count": int(np.count_nonzero(solve_cell_mask)),
            "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
            "phase_target_count": int(len(phase_payload)),
            "boil_target_count": int(len(boil_payload)),
            "condense_target_count": int(len(condense_payload)),
            "solve_tile_mask": solve_tile_mask.tolist(),
            "solve_cell_mask": solve_cell_mask.tolist(),
            "solve_gas_mask": solve_gas_mask.tolist(),
            "cell_changed": bool(snapshot["cell_changed"]),
            "ambient_changed": bool(snapshot["ambient_changed"]),
            "material_changed": bool(snapshot["material_changed"]),
            "phase_changed": bool(snapshot["phase_changed"]),
            "integrity_changed": bool(snapshot["integrity_changed"]),
            "gas_changed": bool(snapshot["gas_changed"]),
            "cell_temperature_range": np.asarray(snapshot["cell_temperature_range"], dtype=np.float32).round(4).tolist(),
            "ambient_temperature_range": np.asarray(snapshot["ambient_temperature_range"], dtype=np.float32).round(4).tolist(),
            "integrity_range": np.asarray(snapshot["integrity_range"], dtype=np.float32).round(4).tolist(),
            "phase_targets": phase_payload,
            "boil_targets": boil_payload,
            "condense_targets": condense_payload,
        }

    def serialize_liquid_runtime(self) -> dict[str, Any]:
        snapshot = self.liquid_solver.runtime_snapshot()
        solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
        post_tile_mask = np.asarray(snapshot["post_tile_mask"], dtype=np.uint8)
        post_cell_mask = np.asarray(snapshot["post_cell_mask"], dtype=np.uint8)
        vertical_seam_mask = np.asarray(snapshot["vertical_seam_mask"], dtype=np.uint8)
        horizontal_seam_mask = np.asarray(snapshot["horizontal_seam_mask"], dtype=np.uint8)
        buoyancy_mask = np.asarray(snapshot["buoyancy_mask"], dtype=np.uint8)
        changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.uint8)
        return {
            "backend": self.liquid_solver.last_backend,
            "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
            "cell_grid_size": [int(self.width), int(self.height)],
            "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
            "post_tile_count": int(np.count_nonzero(post_tile_mask)),
            "post_cell_count": int(np.count_nonzero(post_cell_mask)),
            "vertical_seam_cell_count": int(np.count_nonzero(vertical_seam_mask)),
            "horizontal_seam_cell_count": int(np.count_nonzero(horizontal_seam_mask)),
            "buoyancy_candidate_count": int(np.count_nonzero(buoyancy_mask)),
            "changed_cell_count": int(np.count_nonzero(changed_cell_mask)),
            "material_changed": bool(snapshot["material_changed"]),
            "phase_changed": bool(snapshot["phase_changed"]),
            "velocity_changed": bool(snapshot["velocity_changed"]),
            "temperature_changed": bool(snapshot["temperature_changed"]),
            "integrity_changed": bool(snapshot["integrity_changed"]),
            "placeholder_changed": bool(snapshot["placeholder_changed"]),
            "pending_placeholder_count_before": int(snapshot["pending_placeholder_count_before"]),
            "pending_placeholder_count_after": int(snapshot["pending_placeholder_count_after"]),
            "liquid_cell_count_before": int(snapshot["liquid_cell_count_before"]),
            "liquid_cell_count_after": int(snapshot["liquid_cell_count_after"]),
            "solve_tile_mask": solve_tile_mask.tolist(),
            "post_tile_mask": post_tile_mask.tolist(),
            "post_cell_mask": post_cell_mask.tolist(),
            "vertical_seam_mask": vertical_seam_mask.tolist(),
            "horizontal_seam_mask": horizontal_seam_mask.tolist(),
            "buoyancy_mask": buoyancy_mask.tolist(),
            "changed_cell_mask": changed_cell_mask.tolist(),
        }

    def serialize_reaction_runtime(self) -> dict[str, Any]:
        snapshot = self.reaction_solver.runtime_snapshot()
        stage_tile_masks = {
            stage: np.asarray(mask, dtype=np.uint8)
            for stage, mask in snapshot["stage_tile_masks"].items()
        }
        solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
        changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.uint8)
        changed_gas_mask = np.asarray(snapshot["changed_gas_mask"], dtype=np.uint8)
        ambient_changed_mask = np.asarray(snapshot["ambient_changed_mask"], dtype=np.uint8)
        timer_changed_mask = np.asarray(snapshot["timer_changed_mask"], dtype=np.uint8)
        emitted_light_mask = np.asarray(snapshot["emitted_light_mask"], dtype=np.uint8)
        emitted_material_mask = np.asarray(snapshot["emitted_material_mask"], dtype=np.uint8)
        solve_tile_mask = np.zeros((self.active.tile_height, self.active.tile_width), dtype=np.uint8)
        for mask in stage_tile_masks.values():
            solve_tile_mask |= mask
        return {
            "backend": str(snapshot["backend"]),
            "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
            "cell_grid_size": [int(self.width), int(self.height)],
            "gas_grid_size": [int(self.gas_width), int(self.gas_height)],
            "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
            "solve_cell_count": int(np.count_nonzero(solve_cell_mask)),
            "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
            "changed_cell_count": int(np.count_nonzero(changed_cell_mask)),
            "changed_gas_count": int(np.count_nonzero(changed_gas_mask)),
            "ambient_changed_count": int(np.count_nonzero(ambient_changed_mask)),
            "timer_changed_count": int(np.count_nonzero(timer_changed_mask)),
            "emitted_light_count": int(snapshot["emitted_light_count"]),
            "emitted_material_count": int(snapshot["emitted_material_count"]),
            "executed_action_count": int(snapshot["executed_action_count"]),
            "emit_light_action_count": int(snapshot["emit_light_action_count"]),
            "emit_material_action_count": int(snapshot["emit_material_action_count"]),
            "modify_gas_action_count": int(snapshot["modify_gas_action_count"]),
            "convert_material_action_count": int(snapshot["convert_material_action_count"]),
            "modify_temperature_action_count": int(snapshot["modify_temperature_action_count"]),
            "harm_action_count": int(snapshot["harm_action_count"]),
            "stage_action_counts": {stage: int(count) for stage, count in snapshot["stage_action_counts"].items()},
            "stage_tile_masks": {stage: mask.tolist() for stage, mask in stage_tile_masks.items()},
            "solve_cell_mask": solve_cell_mask.tolist(),
            "solve_gas_mask": solve_gas_mask.tolist(),
            "changed_cell_mask": changed_cell_mask.tolist(),
            "changed_gas_mask": changed_gas_mask.tolist(),
            "ambient_changed_mask": ambient_changed_mask.tolist(),
            "timer_changed_mask": timer_changed_mask.tolist(),
            "emitted_light_mask": emitted_light_mask.tolist(),
            "emitted_material_mask": emitted_material_mask.tolist(),
        }

    def serialize_collapse_runtime(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
        snapshot = self.collapse_solver.runtime_snapshot(
            self,
            allow_gpu_sync_readback=allow_gpu_sync_readback,
        )
        solve_region_mask = np.asarray(snapshot["solve_region_mask"], dtype=np.uint8)
        structural_mask = np.asarray(snapshot["structural_mask"], dtype=np.uint8)
        support_seed_mask = np.asarray(snapshot["support_seed_mask"], dtype=np.uint8)
        supported_mask = np.asarray(snapshot["supported_mask"], dtype=np.uint8)
        unsupported_mask = np.asarray(snapshot["unsupported_mask"], dtype=np.uint8)
        delayed_pending_mask = np.asarray(snapshot["delayed_pending_mask"], dtype=np.uint8)
        immune_unsupported_mask = np.asarray(snapshot["immune_unsupported_mask"], dtype=np.uint8)
        collapsed_cell_mask = np.asarray(snapshot["collapsed_cell_mask"], dtype=np.uint8)
        return {
            "backend": str(snapshot["backend"]),
            "gpu_authoritative": bool(snapshot.get("gpu_authoritative", False)),
            "gpu_authoritative_resources": list(snapshot.get("gpu_authoritative_resources", [])),
            "snapshot_source": str(snapshot.get("snapshot_source", "cpu")),
            "snapshot_stale": bool(snapshot.get("snapshot_stale", False)),
            "gpu_authoritative_snapshot_stale": bool(snapshot.get("gpu_authoritative_snapshot_stale", False)),
            "stale_resources": list(snapshot.get("stale_resources", [])),
            "sync_readback_required": bool(snapshot.get("sync_readback_required", False)),
            "sync_readback_performed": bool(snapshot.get("sync_readback_performed", False)),
            "cell_grid_size": [int(self.width), int(self.height)],
            "dirty_region_count_before": int(snapshot["dirty_region_count_before"]),
            "solve_region_count": int(snapshot["solve_region_count"]),
            "solve_region_cell_count": int(np.count_nonzero(solve_region_mask)),
            "structural_cell_count": int(np.count_nonzero(structural_mask)),
            "support_seed_count": int(np.count_nonzero(support_seed_mask)),
            "supported_cell_count": int(np.count_nonzero(supported_mask)),
            "unsupported_cell_count": int(np.count_nonzero(unsupported_mask)),
            "delayed_pending_count": int(np.count_nonzero(delayed_pending_mask)),
            "immune_unsupported_count": int(np.count_nonzero(immune_unsupported_mask)),
            "collapsed_cell_count": int(np.count_nonzero(collapsed_cell_mask)),
            "collapsed_component_count": int(len(snapshot["collapsed_components"])),
            "solve_region_mask": solve_region_mask.tolist(),
            "structural_mask": structural_mask.tolist(),
            "support_seed_mask": support_seed_mask.tolist(),
            "supported_mask": supported_mask.tolist(),
            "unsupported_mask": unsupported_mask.tolist(),
            "delayed_pending_mask": delayed_pending_mask.tolist(),
            "immune_unsupported_mask": immune_unsupported_mask.tolist(),
            "collapsed_cell_mask": collapsed_cell_mask.tolist(),
            "collapsed_components": [
                {
                    "island_id": int(component["island_id"]),
                    "bbox": (
                        list(component["world_bbox"])
                        if component.get("world_bbox") is not None
                        else list(self._buffer_bbox_to_world_bbox(tuple(int(value) for value in component["bbox"])))
                    ),
                    "cell_count": int(component["cell_count"]),
                }
                for component in snapshot["collapsed_components"]
            ],
        }

    def serialize_optics_runtime(self) -> dict[str, Any]:
        snapshot = self.optics_solver.runtime_snapshot()
        solve_tile_mask = np.asarray(snapshot["solve_tile_mask"], dtype=np.uint8)
        solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.uint8)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.uint8)
        visible_changed_mask = np.asarray(snapshot["visible_changed_mask"], dtype=np.uint8)
        cell_dose_changed_mask = np.asarray(snapshot["cell_dose_changed_mask"], dtype=np.uint8)
        gas_dose_changed_mask = np.asarray(snapshot["gas_dose_changed_mask"], dtype=np.uint8)
        emitter_origin_mask = np.asarray(snapshot["emitter_origin_mask"], dtype=np.uint8)
        public_emitters = snapshot.get("public_emitters")
        emitters_payload = (
            [
                {
                    "light_type": str(emitter["light_type"]),
                    "origin": list(emitter["origin"]),
                    "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
                    "spread": float(emitter["spread"]),
                    "strength": float(emitter["strength"]),
                    "range_cells": int(emitter["range_cells"]),
                }
                for emitter in public_emitters
            ]
            if isinstance(public_emitters, list) and public_emitters
            else [
                {
                    "light_type": str(emitter["light_type"]),
                    "origin": list(
                        self._buffer_to_world_position((int(emitter["origin"][0]), int(emitter["origin"][1])))
                    ),
                    "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
                    "spread": float(emitter["spread"]),
                    "strength": float(emitter["strength"]),
                    "range_cells": int(emitter["range_cells"]),
                }
                for emitter in snapshot["emitters"]
            ]
        )
        return {
            "backend": str(snapshot["backend"]),
            "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
            "cell_grid_size": [int(self.width), int(self.height)],
            "gas_grid_size": [int(self.gas_width), int(self.gas_height)],
            "emitter_count": int(snapshot["emitter_count"]),
            "secondary_branch_count": int(snapshot["secondary_branch_count"]),
            "solve_tile_count": int(np.count_nonzero(solve_tile_mask)),
            "solve_cell_count": int(np.count_nonzero(solve_cell_mask)),
            "solve_gas_count": int(np.count_nonzero(solve_gas_mask)),
            "visible_changed_count": int(np.count_nonzero(visible_changed_mask)),
            "cell_dose_changed_count": int(np.count_nonzero(cell_dose_changed_mask)),
            "gas_dose_changed_count": int(np.count_nonzero(gas_dose_changed_mask)),
            "visible_energy_total": round(float(snapshot["visible_energy_total"]), 4),
            "cell_dose_total": round(float(snapshot["cell_dose_total"]), 4),
            "gas_dose_total": round(float(snapshot["gas_dose_total"]), 4),
            "emitters": emitters_payload,
            "solve_tile_mask": solve_tile_mask.tolist(),
            "solve_cell_mask": solve_cell_mask.tolist(),
            "solve_gas_mask": solve_gas_mask.tolist(),
            "visible_changed_mask": visible_changed_mask.tolist(),
            "cell_dose_changed_mask": cell_dose_changed_mask.tolist(),
            "gas_dose_changed_mask": gas_dose_changed_mask.tolist(),
            "emitter_origin_mask": emitter_origin_mask.tolist(),
        }

    def serialize_active_runtime(self) -> dict[str, Any]:
        active_tile_ttl = np.asarray(self.active.active_tile_ttl, dtype=np.int32)
        active_chunk_mask = np.asarray(self.active.active_chunk_mask, dtype=np.uint8)
        ys, xs = np.nonzero(self.placeholder_displaced_material > 0)
        pending_displaced_cells: list[dict[str, Any]] = []
        for buffer_y, buffer_x in zip(ys.tolist(), xs.tolist()):
            world_x, world_y = self._buffer_to_world_position((buffer_x, buffer_y))
            pending_displaced_cells.append(
                {
                    "x": int(world_x),
                    "y": int(world_y),
                    "material_id": int(self.placeholder_displaced_material[buffer_y, buffer_x]),
                }
            )
        pending_displaced_cells.sort(key=lambda cell: (int(cell["y"]), int(cell["x"])))
        return {
            "tile_size": int(self.active.tile_size),
            "chunk_tiles": int(self.active.chunk_tiles),
            "active_ttl_reset": int(self.active.active_ttl_reset),
            "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
            "chunk_grid_size": [int(self.active.chunk_width), int(self.active.chunk_height)],
            "active_tile_count": int(np.count_nonzero(active_tile_ttl > 0)),
            "active_chunk_count": int(np.count_nonzero(active_chunk_mask > 0)),
            "active_tile_ttl": active_tile_ttl.tolist(),
            "active_chunk_mask": active_chunk_mask.tolist(),
            "pending_displaced_count": int(len(pending_displaced_cells)),
            "pending_displaced_cells": pending_displaced_cells,
        }

    def serialize_motion_runtime(self) -> dict[str, Any]:
        snapshot = self.motion_solver.runtime_snapshot()
        public_powder_reservations = snapshot.get("public_powder_reservations")
        if isinstance(public_powder_reservations, list) and public_powder_reservations:
            powder_payload: list[dict[str, Any]] = [dict(record) for record in public_powder_reservations]
        else:
            powder_payload = []
            for record in snapshot["powder_reservations"]:
                item: dict[str, Any] = {}
                for name in snapshot["powder_reservations"].dtype.names or ():
                    value = record[name]
                    if isinstance(value, np.ndarray):
                        if name in {"source_xy", "desired_target_xy", "reserved_target_xy", "resolved_target_xy"}:
                            world_x, world_y = self._buffer_to_world_position((int(value[0]), int(value[1])))
                            item[name] = [int(world_x), int(world_y)]
                        else:
                            item[name] = value.tolist()
                    elif isinstance(value, np.generic):
                        item[name] = value.item()
                    else:
                        item[name] = value
                powder_payload.append(item)
        public_island_reservations = snapshot.get("public_island_reservations")
        if isinstance(public_island_reservations, list) and public_island_reservations:
            island_payload: list[dict[str, Any]] = [dict(record) for record in public_island_reservations]
        else:
            island_payload = []
            for record in snapshot["island_reservations"]:
                item = {}
                for name in snapshot["island_reservations"].dtype.names or ():
                    value = record[name]
                    if name == "buffer_bbox":
                        item["world_bbox"] = list(
                            self._buffer_bbox_to_world_bbox(tuple(int(component) for component in np.asarray(value).tolist()))
                        )
                        continue
                    if isinstance(value, np.ndarray):
                        item[name] = value.tolist()
                    elif isinstance(value, np.generic):
                        item[name] = value.item()
                    else:
                        item[name] = value
                island_payload.append(item)
        return {
            "backend": self.motion_solver.last_backend,
            "powder_reservation_count": int(snapshot["powder_reservations"].shape[0]),
            "island_reservation_count": int(snapshot["island_reservations"].shape[0]),
            "powder_reservations": powder_payload,
            "island_reservations": island_payload,
        }

    def serialize_paging_state(self) -> dict[str, Any]:
        x0, y0, x1, y1 = self.paging.active_bounds()
        return {
            "origin": [self.paging.origin_x, self.paging.origin_y],
            "buffer_origin": [self.paging.buffer_origin_x, self.paging.buffer_origin_y],
            "active_bounds": [x0, y0, x1, y1],
            "buffer_size": [self.width, self.height],
            "active_size": [self.paging.active_width, self.paging.active_height],
            "stored_stripes": self.page_store.stored_count(),
        }

    def serialize_engine_capabilities(self) -> dict[str, Any]:
        material_payload = self._shadow_material_payload()
        gas_payload = self._shadow_gas_species_payload()
        light_payload = self._shadow_light_type_payload()
        material_name_aliases = deepcopy(BASE_MATERIAL_RUNTIME_ALIASES)
        force_source_fields = ["x", "y", "direction", "radius", "strength", "lifetime"]
        emitter_fields = ["x", "y", "light_type", "strength", "radius", "direction", "spread"]
        entity_state_fields = [
            "entity_id",
            "x",
            "y",
            "width",
            "height",
            "velocity_xy",
            "facing_xy",
            "placeholder_material",
            "tags",
            "observe_channels",
            "observe_pad_cells",
            "observe_width",
            "observe_height",
            "observe_label",
        ]
        entity_state_patch_fields = ["entity_id", "fields"]
        entity_observation_spec_fields = [
            "entity_id",
            "observe_channels",
            "observe_pad_cells",
            "observe_width",
            "observe_height",
            "observe_label",
        ]
        entity_placeholder_fields = ["entity_id", "x", "y", "width", "height", "material"]
        target_query_fields = [
            "query_id",
            "anchor_filters",
            "source_entity_id",
            "source_x",
            "source_y",
            "anchor_entity_id",
            "direction",
            "distance_cells",
            "distance_meters",
            "distance_hint",
            "require_empty",
            "search_radius",
            "label",
        ]
        target_query_overlay_fields = ["target_query_id", "target_dx", "target_dy"]
        inline_target_query_optional_fields = [*target_query_overlay_fields, "target_queries"]
        readback_request_fields = [
            "request_id",
            "center_x",
            "center_y",
            "width",
            "height",
            "channels",
            "observer_id",
            "label",
            "target_query_id",
            "target_dx",
            "target_dy",
        ]
        observation_target_fields = [
            "observer_id",
            "channels",
            "center_x",
            "center_y",
            "width",
            "height",
            "entity_id",
            "pad_cells",
            "label",
            "target_query_id",
            "target_dx",
            "target_dy",
        ]
        change_intent_fields = [
            "intent_id",
            "target_query_id",
            "center_x",
            "center_y",
            "target_dx",
            "target_dy",
            "radius",
            "material",
            "temperature_delta",
            "velocity",
            "velocity_carrier",
            "velocity_mode",
            "require_empty",
            "fallback_mode",
            "fallback_radius",
            "potency",
            "stability",
            "label",
        ]
        carrier_intent_fields = [
            "intent_id",
            "kind",
            "target_query_id",
            "center_x",
            "center_y",
            "source_entity_id",
            "source_x",
            "source_y",
            "target_dx",
            "target_dy",
            "radius",
            "material",
            "gas_species",
            "gas_amount",
            "light_type",
            "light_strength",
            "light_spread",
            "force_radius",
            "force_strength",
            "force_lifetime",
            "release_mode",
            "require_empty",
            "fallback_mode",
            "fallback_radius",
            "potency",
            "stability",
            "label",
        ]
        material_fields = [
            "material_id",
            "name",
            "display_name",
            "default_phase",
            "render_group",
            "base_color",
            "density",
            "gravity_scale",
            "wind_coupling",
            "drag_scale",
            "friction",
            "elasticity",
            "max_dda_step",
            "powder_solver_kind",
            "liquid_solver_kind",
            "falling_island_break_kind",
            "is_structural",
            "is_support_anchor",
            "collapse_behavior",
            "collapse_generation",
            "powder_generation",
            "base_integrity",
            "heat_capacity",
            "conductivity",
            "ambient_exchange_rate",
            "melt_point",
            "boil_point",
            "melt_to_material",
            "freeze_to_material",
            "boil_to_gas_species",
            "spawn_temperature",
            "material_tag_mask",
            "gas_tag_mask",
            "light_tag_mask",
            "reaction_slots",
            "tags",
        ]
        gas_fields = [
            "species_id",
            "name",
            "display_name",
            "color",
            "diffusion_rate",
            "buoyancy",
            "decay_rate",
            "temperature_coupling",
            "condense_point",
            "condense_to_material",
            "pressure_factor",
            "density_factor",
            "material_reaction_tag_mask",
            "light_reaction_tag_mask",
        ]
        light_fields = [
            "light_type_id",
            "name",
            "display_name",
            "color",
            "visual_channel",
            "default_range",
            "max_bounce",
            "dose_channel_id",
            "render_style",
        ]
        optics_fields = ["material_name", "light_type", "absorption", "scattering", "refraction"]
        reaction_action_fields = [
            "reaction_type",
            "target_material",
            "emit_material",
            "light_type",
            "gas_species",
            "duration",
            "speed",
            "velocity",
            "direction",
            "strength",
            "beam_width",
            "range_cells",
            "delta",
            "value",
            "generation",
            "harm_per_frame",
            "integrity_threshold",
            "allow_subunit_scale",
        ]
        reaction_table_rule_sets = [
            "material_material",
            "material_gas",
            "material_light",
            "gas_gas",
            "gas_light",
            "self_rules",
        ]
        pair_reaction_rule_fields = [
            "lhs_material",
            "lhs_gas",
            "rhs_material",
            "rhs_gas",
            "rhs_light",
            "lhs_tag_mask",
            "rhs_tag_mask",
            "phases",
            "min_temperature",
            "max_temperature",
            "threshold",
            "rate",
            "consume_policy",
            "result_action",
            "trigger_slot_index",
        ]
        self_reaction_rule_fields = [
            "material",
            "trigger_slot_index",
            "phases",
            "min_temperature",
            "max_temperature",
            "timer_index",
            "integrity_at_most",
            "integrity_at_least",
        ]
        cell_window_fields = [
            "origin",
            "size",
            "material_id",
            "phase",
            "cell_flags",
            "velocity",
            "cell_temperature",
            "temperature",
            "timer_pack",
            "integrity",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_delay_pending",
        ]
        temperature_window_fields = ["temperature"]
        gas_window_fields = ["species", "concentration"]
        pressure_window_fields = ["pressure"]
        velocity_window_fields = ["velocity"]
        optics_window_fields = ["origin", "size", "gas_origin", "gas_size", "visible_illumination", "cell_dose", "gas_dose"]
        visible_illumination_fields = ["illumination"]
        debug_frame_fields = ["view", "origin", "size", "gas_species", "light_type", "frame"]
        bridge_runtime_fields = [
            "frame_id",
            "bridge",
            "pending_readbacks",
            "inflight_readbacks",
            "ready_readbacks",
            "pending_commands",
        ]
        bridge_resource_catalog_fields = ["typed_tables", "shadow_buffers", "snapshots"]
        bridge_typed_table_fields = ["name", "shape", "dtype", "structured", "field_names", "row_count", "rows"]
        bridge_typed_table_slice_fields = [
            "name",
            "shape",
            "dtype",
            "structured",
            "field_names",
            "row_count",
            "offset",
            "limit",
            "returned_count",
            "slice_shape",
            "rows",
        ]
        bridge_shadow_buffer_fields = ["name", "shape", "dtype", "structured", "field_names", "row_count", "rows", "values", "utf8"]
        bridge_shadow_buffer_slice_fields = [
            "name",
            "shape",
            "dtype",
            "structured",
            "field_names",
            "row_count",
            "offset",
            "limit",
            "returned_count",
            "slice_shape",
            "rows",
            "values",
            "utf8",
        ]
        bridge_shadow_buffer_window_fields = [
            "name",
            "shape",
            "dtype",
            "structured",
            "field_names",
            "row_count",
            "window_origin",
            "requested_size",
            "window_size",
            "window_axes",
            "returned_shape",
            "rows",
            "values",
            "utf8",
        ]
        bridge_shadow_buffer_spatial_window_fields = [
            "name",
            "shape",
            "dtype",
            "structured",
            "field_names",
            "row_count",
            "coord_space",
            "window_origin",
            "requested_size",
            "window_size",
            "window_axes",
            "returned_shape",
            "rows",
            "values",
            "utf8",
        ]
        bridge_upload_snapshot_fields = ["frame_meta", "world_commands", "readback_requests", "page_stripes"]
        bridge_readback_stage_fields = ["request_id", "stage"]
        bridge_index_stage_fields = ["index", "stage"]
        world_command_material_alias_fields_by_kind = {
            "inject_material": ["material"],
            "write_material_region": ["material"],
            "sync_entity_states": ["entities[].placeholder_material"],
            "patch_entity_states": ["patches[].fields.placeholder_material"],
        }
        world_command_collection_item_types_by_kind = {
            "sync_entity_states": {"entities": "entity_state"},
            "patch_entity_states": {"patches": "entity_state_patch"},
            "sync_entity_observation_specs": {"observations": "entity_observation_spec"},
        }
        bridge_frame_snapshot_fields = [
            "prepared",
            "commands",
            "command_stages",
            "readback_requests",
            "readback_request_stages",
            "placeholders",
            "placeholder_stages",
            "placeholder_dirty_rects",
            "paging_updates",
            "paging_update_stages",
            "page_stripes",
            "page_stripe_stages",
        ]
        readback_state_fields = [
            "queued",
            "queued_commands",
            "pending",
            "pending_requests",
            "inflight",
            "inflight_requests",
            "ready",
        ]
        readback_result_fields = ["frame_id", "request", "payload"]
        readback_poll_fields = ["ready", "status", "result"]
        readback_plan_fields = ["request", "layout", "nbytes", "gpu_source_count", "cpu_chunk_count", "payload"]
        observation_plan_fields = ["target", *readback_plan_fields]
        readback_channel_payload_types = {
            "cell": "readback_cell_payload",
            "ambient_temperature": "readback_scalar_window",
            "pressure": "readback_scalar_window",
            "velocity": "readback_vector_window",
            "optics": "readback_optics_payload",
            "gas": "readback_gas_payload",
        }
        readback_payload_fields = list(READBACK_ALLOWED_CHANNELS)
        readback_cell_payload_fields = [
            "origin",
            "size",
            "core_words",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_delay_pending",
        ]
        readback_scalar_window_fields = ["origin", "size", "grid", "values"]
        readback_vector_window_fields = ["origin", "size", "grid", "values"]
        readback_gas_payload_fields = ["origin", "size", "grid", "species"]
        readback_optics_payload_fields = [
            "origin",
            "size",
            "gas_origin",
            "gas_size",
            "visible_illumination",
            "cell_dose",
            "gas_dose",
        ]
        resolved_target_fields = [
            "query_id",
            "status",
            "anchor_filters",
            "direction",
            "distance_cells",
            "distance_meters",
            "distance_hint",
            "label",
            "source_position",
            "source_world_position",
            "anchor_kind",
            "anchor_entity_id",
            "anchor_position",
            "anchor_world_position",
            "resolved_position",
            "resolved_world_position",
            "note",
        ]
        resolved_change_intent_fields = [
            "intent_id",
            "status",
            "target_query_id",
            "label",
            "potency",
            "stability",
            "center_position",
            "center_world_position",
            "effective_radius",
            "material",
            "temperature_delta",
            "velocity",
            "velocity_carrier",
            "velocity_mode",
            "require_empty",
            "fallback_mode",
            "fallback_applied",
            "effect_shape",
            "effect_cells",
            "effect_bounds",
            "generated_commands",
            "note",
        ]
        resolved_carrier_intent_fields = [
            "intent_id",
            "status",
            "kind",
            "target_query_id",
            "label",
            "release_mode",
            "potency",
            "stability",
            "source_position",
            "source_world_position",
            "impact_position",
            "impact_world_position",
            "effective_radius",
            "material",
            "gas_species",
            "gas_amount",
            "light_type",
            "light_strength",
            "light_spread",
            "force_radius",
            "force_strength",
            "force_lifetime",
            "direction",
            "require_empty",
            "fallback_mode",
            "fallback_applied",
            "effect_shape",
            "effect_cells",
            "effect_bounds",
            "generated_commands",
            "note",
        ]
        observation_result_fields = ["observer_id", "frame_id", "request", "payload"]
        emitter_runtime_fields = ["persistent_emitters", "queued_emitters"]
        paging_state_fields = ["origin", "buffer_origin", "active_bounds", "buffer_size", "active_size", "stored_stripes"]
        pending_commands_fields = ["pending", "commands"]
        entity_observation_runtime_fields = ["observations", "targets", "requests"]
        entity_observation_consume_result_fields = [
            "frame_id",
            "consumed",
            "consumed_readbacks",
            "observations",
            "entity_feedback",
        ]
        control_state_fields = ["paused", "speed", "single_step"]
        readback_status_fields = ["request_id", "status"]
        readback_cancel_request_fields = ["request_id"]
        readback_cancel_result_fields = ["ok", "request_id", "status"]
        frame_pending_state_fields = ["pending", "submission_ids"]
        frame_state_fields = [
            "pending",
            "pending_submission_ids",
            "ready",
            "ready_submission_ids",
            "canceled_submission_ids",
        ]
        frame_submission_status_fields = ["submission_id", "status"]
        frame_cancel_request_fields = ["submission_id"]
        frame_cancel_result_fields = ["ok", "submission_id", "status", "pending_frames"]
        frame_cancel_all_result_fields = ["ok", "canceled_submission_ids", "pending_frames"]
        deferred_readback_request_result_fields = ["ok", "queued", "pending_commands", "request_id"]
        deferred_observation_request_result_fields = ["ok", "queued", "pending_commands", "request_id"]
        deferred_frame_submit_ack_fields = ["ok", "queued", "pending_frames", "submission_id"]
        deferred_controller_state_result_fields = ["ok", "queued", "pending_frames", "submission_id"]
        material_fill_request_fields = [
            "x",
            "y",
            "width",
            "height",
            "material",
            "immediate",
            *inline_target_query_optional_fields,
        ]
        paging_focus_request_fields = ["x", "y", *inline_target_query_optional_fields]
        paging_focus_result_fields = ["ok", "queued", "pending_commands", "target_center"]
        page_store_has_result_fields = ["stored"]
        page_store_apply_result_fields = ["ok", "stored", "queued", "pending_commands"]
        page_store_clear_result_fields = ["ok", "cleared", "stored_stripes"]
        page_stripe_apply_result_fields = ["ok", "queued", "pending_commands"]
        speed_request_fields = ["speed"]
        pause_result_fields = ["paused"]
        resume_result_fields = ["paused"]
        step_result_fields = ["single_step"]
        speed_result_fields = ["speed"]
        queued_mutation_result_fields = ["ok", "queued", "pending_commands"]
        control_reset_result_fields = ["ok", "queued", "pending_commands"]
        material_injector_fields = ["x", "y", "material", "radius", "immediate", *inline_target_query_optional_fields]
        temperature_injector_fields = ["x", "y", "delta", "radius", "immediate", *inline_target_query_optional_fields]
        velocity_injector_fields = [
            "x",
            "y",
            "velocity",
            "radius",
            "carrier",
            "mode",
            "immediate",
            *inline_target_query_optional_fields,
        ]
        gas_injector_fields = ["x", "y", "species", "amount", "radius", "immediate", *inline_target_query_optional_fields]
        force_injector_fields = [*force_source_fields, "immediate", *inline_target_query_optional_fields]
        light_injector_fields = [*emitter_fields, "immediate", *inline_target_query_optional_fields]
        material_patch_fields = ["name", "fields"]
        gas_patch_fields = ["name", "fields"]
        light_patch_fields = ["name", "fields"]
        optics_patch_fields = ["material_name", "light_type", "fields"]
        reaction_action_patch_fields = ["index", "fields"]
        reaction_action_delete_request_fields = ["index"]
        reaction_rule_patch_fields = ["rule_set", "index", "fields"]
        reaction_rule_delete_request_fields = ["rule_set", "index"]
        materials_table_fields = ["materials"]
        gases_table_fields = ["gases"]
        lights_table_fields = ["lights"]
        optics_table_fields = ["optics"]
        reactions_table_fields = ["actions", "rules"]
        gas_species_runtime_fields = ["species_id", "species", "total_concentration", "active_concentration"]
        heat_phase_target_fields = ["x", "y", "target_material_id", "target_material"]
        heat_boil_target_fields = ["x", "y", "target_species_id", "target_species"]
        heat_condense_target_fields = [
            "gas_x",
            "gas_y",
            "species_id",
            "species",
            "target_material_id",
            "target_material",
        ]
        collapse_component_fields = ["island_id", "bbox", "cell_count"]
        pending_displaced_cell_fields = ["x", "y", "material_id"]
        powder_reservation_fields = [
            "source_xy",
            "desired_target_xy",
            "reserved_target_xy",
            "resolved_target_xy",
            "velocity_xy",
            "material_id",
            "resolve_state",
        ]
        island_reservation_fields = [
            "island_id",
            "world_bbox",
            "velocity_xy",
            "subcell_offset",
            "target_shift",
            "reserved_shift",
            "resolved_shift",
            "resolve_state",
        ]
        gas_runtime_fields = [
            "backend",
            "pressure_iterations",
            "tile_grid_size",
            "gas_grid_size",
            "solve_tile_count",
            "solve_gas_count",
            "solve_tile_mask",
            "solve_gas_mask",
            "force_source_count_before",
            "force_source_count_after",
            "velocity_changed",
            "ambient_changed",
            "gas_changed",
            "pressure_range",
            "ambient_range",
            "flow_speed_range",
            "species_runtime",
        ]
        heat_runtime_fields = [
            "backend",
            "ambient_iterations",
            "tile_grid_size",
            "cell_grid_size",
            "gas_grid_size",
            "solve_tile_count",
            "solve_cell_count",
            "solve_gas_count",
            "phase_target_count",
            "boil_target_count",
            "condense_target_count",
            "solve_tile_mask",
            "solve_cell_mask",
            "solve_gas_mask",
            "cell_changed",
            "ambient_changed",
            "material_changed",
            "phase_changed",
            "integrity_changed",
            "gas_changed",
            "cell_temperature_range",
            "ambient_temperature_range",
            "integrity_range",
            "phase_targets",
            "boil_targets",
            "condense_targets",
        ]
        liquid_runtime_fields = [
            "backend",
            "tile_grid_size",
            "cell_grid_size",
            "solve_tile_count",
            "post_tile_count",
            "post_cell_count",
            "vertical_seam_cell_count",
            "horizontal_seam_cell_count",
            "buoyancy_candidate_count",
            "changed_cell_count",
            "material_changed",
            "phase_changed",
            "velocity_changed",
            "temperature_changed",
            "integrity_changed",
            "placeholder_changed",
            "pending_placeholder_count_before",
            "pending_placeholder_count_after",
            "liquid_cell_count_before",
            "liquid_cell_count_after",
            "solve_tile_mask",
            "post_tile_mask",
            "post_cell_mask",
            "vertical_seam_mask",
            "horizontal_seam_mask",
            "buoyancy_mask",
            "changed_cell_mask",
        ]
        reaction_runtime_fields = [
            "backend",
            "tile_grid_size",
            "cell_grid_size",
            "gas_grid_size",
            "solve_tile_count",
            "solve_cell_count",
            "solve_gas_count",
            "changed_cell_count",
            "changed_gas_count",
            "ambient_changed_count",
            "timer_changed_count",
            "emitted_light_count",
            "emitted_material_count",
            "executed_action_count",
            "emit_light_action_count",
            "emit_material_action_count",
            "modify_gas_action_count",
            "convert_material_action_count",
            "modify_temperature_action_count",
            "harm_action_count",
            "stage_action_counts",
            "stage_tile_masks",
            "solve_cell_mask",
            "solve_gas_mask",
            "changed_cell_mask",
            "changed_gas_mask",
            "ambient_changed_mask",
            "timer_changed_mask",
            "emitted_light_mask",
            "emitted_material_mask",
        ]
        collapse_runtime_fields = [
            "backend",
            "cell_grid_size",
            "dirty_region_count_before",
            "solve_region_count",
            "solve_region_cell_count",
            "structural_cell_count",
            "support_seed_count",
            "supported_cell_count",
            "unsupported_cell_count",
            "delayed_pending_count",
            "immune_unsupported_count",
            "collapsed_cell_count",
            "collapsed_component_count",
            "solve_region_mask",
            "structural_mask",
            "support_seed_mask",
            "supported_mask",
            "unsupported_mask",
            "delayed_pending_mask",
            "immune_unsupported_mask",
            "collapsed_cell_mask",
            "collapsed_components",
        ]
        optics_runtime_fields = [
            "backend",
            "tile_grid_size",
            "cell_grid_size",
            "gas_grid_size",
            "emitter_count",
            "secondary_branch_count",
            "solve_tile_count",
            "solve_cell_count",
            "solve_gas_count",
            "visible_changed_count",
            "cell_dose_changed_count",
            "gas_dose_changed_count",
            "visible_energy_total",
            "cell_dose_total",
            "gas_dose_total",
            "emitters",
            "solve_tile_mask",
            "solve_cell_mask",
            "solve_gas_mask",
            "visible_changed_mask",
            "cell_dose_changed_mask",
            "gas_dose_changed_mask",
            "emitter_origin_mask",
        ]
        active_runtime_fields = [
            "tile_size",
            "chunk_tiles",
            "active_ttl_reset",
            "tile_grid_size",
            "chunk_grid_size",
            "active_tile_count",
            "active_chunk_count",
            "active_tile_ttl",
            "active_chunk_mask",
            "pending_displaced_count",
            "pending_displaced_cells",
        ]
        motion_runtime_fields = ["backend", "powder_reservation_count", "island_reservation_count", "powder_reservations", "island_reservations"]
        cell_core_layout_fields = ["word0", "word1", "word2", "word3", "word4"]
        cell_core_unpack_fields = [
            "material_id",
            "phase",
            "cell_flags",
            "velocity",
            "cell_temperature",
            "timer_pack",
            "integrity",
        ]
        public_world_command_kinds = [*PUBLIC_WORLD_COMMAND_KINDS]
        frame_input_field_order = [
            "submission_id",
            "focus_center",
            "controller_state",
            "controller_state_provided",
            "entities",
            "entity_placeholders",
            "force_sources",
            "emitters",
            "target_queries",
            "change_intents",
            "carrier_intents",
            "observation_targets",
            "readback_requests",
            "commands",
        ]
        pending_frame_detail_field_order = [*frame_input_field_order, "preview"]
        frame_preview_field_order = [
            "controller_state",
            "resolved_targets",
            "resolved_change_intents",
            "resolved_carrier_intents",
            "resolved_commands",
            "observation_requests",
            "observation_plans",
            "readback_requests",
            "readback_plans",
            "bridge_frame_snapshot",
            "paging_updates",
            "placeholder_count",
        ]
        frame_output_field_order = [
            "frame_id",
            "submission_id",
            "controller_state",
            "consumed_readbacks",
            "resolved_targets",
            "resolved_change_intents",
            "resolved_carrier_intents",
            "observations",
            "entity_feedback",
            "paging_updates",
            "observation_plans",
            "readback_plans",
            "bridge_upload_snapshot",
            "bridge_frame_snapshot",
            "queued_observations",
            "queued_readbacks",
            "queued_commands",
            "placeholder_count",
        ]
        allowed_gas_species = [
            str(item["name"])
            for item in gas_payload
            if item.get("name")
        ]
        allowed_lights = [
            str(item["name"])
            for item in light_payload
            if item.get("name")
        ]
        debug_view_parameterized_views = {
            "gas": {
                "query_fields": ["gas_species"],
                "default_gas_species": "water_gas",
                "allowed_gas_species": allowed_gas_species,
            },
            "optics": {
                "query_fields": ["light"],
                "light_query_maps_to": "light_type",
                "light_optional": True,
                "omitted_light_means": "all_lights",
                "allowed_lights": allowed_lights,
            },
            "light": {
                "query_fields": ["light"],
                "light_query_maps_to": "light_type",
                "light_optional": True,
                "omitted_light_means": "all_lights",
                "allowed_lights": allowed_lights,
            },
        }
        debug_view_options = {
            "default": self.default_debug_view.value,
            "frame_endpoint": "/api/read/debug_frame",
            "parameterized_views": debug_view_parameterized_views,
        }
        gas_read_options = {
            "species_query_field": "species",
            "default_species": "water_gas",
            "allowed_species": allowed_gas_species,
        }
        optics_read_options = {
            "light_query_field": "light",
            "light_query_maps_to": "light_type",
            "light_optional": True,
            "omitted_light_means": "all_lights",
            "allowed_lights": allowed_lights,
        }
        http_api_endpoints = {
            "/api/meta/capabilities": {
                "method": "GET",
                "response_type": "engine_capabilities",
            },
            "/api/read/cells": {
                "method": "GET",
                "query_fields": ["x", "y", "w", "h"],
                "response_type": "cell_window",
                "query_coord_space": "world",
                "default_origin": [0, 0],
                "default_size": [16, 16],
            },
            "/api/read/temperature": {
                "method": "GET",
                "query_fields": ["x", "y", "w", "h"],
                "response_type": "temperature_window",
                "query_coord_space": "world",
                "default_origin": [0, 0],
                "omitted_size_means": "full_buffer",
            },
            "/api/read/gas": {
                "method": "GET",
                "query_fields": ["species"],
                "response_type": "gas_window",
                "coord_space": "gas",
                "coverage": "full_buffer",
                **deepcopy(gas_read_options),
            },
            "/api/read/pressure": {
                "method": "GET",
                "response_type": "pressure_window",
                "coord_space": "gas",
                "coverage": "full_buffer",
            },
            "/api/read/gas_runtime": {
                "method": "GET",
                "response_type": "gas_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/heat_runtime": {
                "method": "GET",
                "response_type": "heat_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/liquid_runtime": {
                "method": "GET",
                "response_type": "liquid_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/reaction_runtime": {
                "method": "GET",
                "response_type": "reaction_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/collapse_runtime": {
                "method": "GET",
                "response_type": "collapse_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/optics_runtime": {
                "method": "GET",
                "response_type": "optics_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/light": {
                "method": "GET",
                "response_type": "visible_illumination",
                "light_type": "visible_light",
                "coord_space": "world",
                "coverage": "full_buffer",
            },
            "/api/read/debug_frame": {
                "method": "GET",
                "query_fields": ["view", "gas_species", "light"],
                "response_type": "debug_frame",
                "default_view": self.default_debug_view.value,
                "allowed_views": [view.value for view in DebugView],
                "parameterized_views": deepcopy(debug_view_parameterized_views),
            },
            "/api/read/optics": {
                "method": "GET",
                "query_fields": ["x", "y", "w", "h", "light"],
                "response_type": "optics_window",
                "query_coord_space": "world",
                "default_origin": [0, 0],
                "omitted_size_means": "full_buffer",
                **deepcopy(optics_read_options),
            },
            "/api/read/velocity": {
                "method": "GET",
                "response_type": "velocity_window",
                "coord_space": "gas",
                "coverage": "full_buffer",
            },
            "/api/read/forces": {
                "method": "GET",
                "response_type": "force_sources_response",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "current_persistent_state",
            },
            "/api/read/emitters": {
                "method": "GET",
                "response_type": "emitters",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "current_persistent_state",
            },
            "/api/read/active": {
                "method": "GET",
                "response_type": "active_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "current_runtime_state",
            },
            "/api/read/motion": {
                "method": "GET",
                "response_type": "motion_runtime",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "latest_solver_runtime",
            },
            "/api/read/bridge_runtime": {
                "method": "GET",
                "response_type": "bridge_runtime",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_runtime_state",
            },
            "/api/read/bridge_resources": {
                "method": "GET",
                "response_type": "bridge_resource_catalog",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_resource_catalog",
            },
            "/api/read/bridge_typed_table": {
                "method": "GET",
                "query_fields": ["name"],
                "response_type": "bridge_typed_table_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_typed_table_snapshot",
            },
            "/api/read/bridge_typed_table_slice": {
                "method": "GET",
                "query_fields": ["name", "offset", "limit"],
                "response_type": "bridge_typed_table_slice_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_typed_table_snapshot",
                "default_offset": 0,
                "default_limit": 64,
            },
            "/api/read/bridge_shadow_buffer": {
                "method": "GET",
                "query_fields": ["name"],
                "response_type": "bridge_shadow_buffer_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_shadow_buffer_snapshot",
            },
            "/api/read/bridge_shadow_buffer_slice": {
                "method": "GET",
                "query_fields": ["name", "offset", "limit"],
                "response_type": "bridge_shadow_buffer_slice_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_shadow_buffer_snapshot",
                "default_offset": 0,
                "default_limit": 64,
            },
            "/api/read/bridge_shadow_buffer_window": {
                "method": "GET",
                "query_fields": ["name", "x", "y", "w", "h"],
                "response_type": "bridge_shadow_buffer_window_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_shadow_buffer_snapshot",
                "query_coord_space": "buffer",
                "default_origin": [0, 0],
                "default_size": [16, 16],
            },
            "/api/read/bridge_shadow_buffer_world_window": {
                "method": "GET",
                "query_fields": ["name", "x", "y", "w", "h"],
                "response_type": "bridge_shadow_buffer_world_window_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_shadow_buffer_snapshot",
                "query_coord_space": "world",
                "default_origin": [0, 0],
                "default_size": [16, 16],
            },
            "/api/read/bridge_shadow_buffer_gas_window": {
                "method": "GET",
                "query_fields": ["name", "x", "y", "w", "h"],
                "response_type": "bridge_shadow_buffer_gas_window_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "current_shadow_buffer_snapshot",
                "query_coord_space": "gas",
                "default_origin": [0, 0],
                "default_size": [4, 4],
            },
            "/api/read/bridge_uploads": {
                "method": "GET",
                "response_type": "bridge_upload_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "latest_upload_snapshot",
            },
            "/api/read/bridge_frame": {
                "method": "GET",
                "response_type": "bridge_frame_snapshot",
                "snapshot_scope": "current_bridge_state",
                "snapshot_kind": "latest_frame_snapshot",
            },
            "/api/read/paging": {
                "method": "GET",
                "response_type": "paging_state",
                "snapshot_scope": "current_world_state",
                "snapshot_kind": "current_paging_state",
            },
            "/api/control/state": {
                "method": "GET",
                "response_fields": ["paused", "speed", "single_step"],
                "response_type": "control_state",
            },
            "/api/commands/pending": {
                "method": "GET",
                "response_fields": ["pending", "commands"],
                "response_type": "pending_commands",
            },
            "/api/targets/preview": {
                "method": "POST",
                "body_fields": ["target_queries"],
                "request_type": "target_query[]",
                "response_fields": ["ok", "resolved_targets"],
                "response_type": "target_preview_result",
            },
            "/api/commands/preview": {
                "method": "POST",
                "body_fields": ["command", "target_queries"],
                "request_type": "world_command",
                "response_fields": ["ok", "command"],
                "response_type": "command_preview_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/commands/request": {
                "method": "POST",
                "body_fields": ["command", "target_queries"],
                "request_type": "world_command",
                "response_fields": ["ok", "queued", "pending_commands", "command"],
                "response_type": "command_request_result",
                "queueing": "deferred_command",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/change_intents/preview": {
                "method": "POST",
                "body_fields": ["intent", "target_queries"],
                "request_type": "change_intent",
                "response_fields": ["ok", "resolved_intent"],
                "response_type": "change_intent_preview_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/change_intents/request": {
                "method": "POST",
                "body_fields": ["intent", "target_queries"],
                "request_type": "change_intent",
                "response_fields": ["ok", "queued", "pending_commands", "resolved_intent"],
                "response_type": "change_intent_request_result",
                "queueing": "deferred_command",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/carrier_intents/preview": {
                "method": "POST",
                "body_fields": ["intent", "target_queries"],
                "request_type": "carrier_intent",
                "response_fields": ["ok", "resolved_intent"],
                "response_type": "carrier_intent_preview_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/carrier_intents/request": {
                "method": "POST",
                "body_fields": ["intent", "target_queries"],
                "request_type": "carrier_intent",
                "response_fields": ["ok", "queued", "pending_commands", "resolved_intent"],
                "response_type": "carrier_intent_request_result",
                "queueing": "deferred_command",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/readback/pending": {
                "method": "GET",
                "response_fields": [
                    "queued",
                    "queued_commands",
                    "pending",
                    "pending_requests",
                    "inflight",
                    "inflight_requests",
                    "ready",
                ],
                "response_type": "readback_state",
            },
            "/api/readback/ready": {
                "method": "GET",
                "response_fields": ["ready", "results"],
                "response_type": "readback_ready",
            },
            "/api/readback/poll": {
                "method": "GET",
                "query_fields": ["request_id"],
                "response_fields": ["ready", "status", "result"],
                "response_type": "readback_poll",
            },
            "/api/readback/poll_all": {
                "method": "GET",
                "response_fields": ["results"],
                "response_type": "readback_poll_all",
            },
            "/api/readback/status": {
                "method": "GET",
                "query_fields": ["request_id"],
                "response_fields": ["request_id", "status"],
                "response_type": "readback_status",
            },
            "/api/readback/cancel": {
                "method": "POST",
                "body_fields": ["request_id"],
                "response_fields": ["ok", "request_id", "status"],
                "request_type": "readback_cancel_request",
                "response_type": "readback_cancel_result",
            },
            "/api/frame/pending": {
                "method": "GET",
                "response_fields": ["pending", "submission_ids"],
                "response_type": "frame_pending_state",
            },
            "/api/frame/pending/detail": {
                "method": "GET",
                "response_fields": ["pending", "frames"],
                "response_type": "pending_frame_details",
            },
            "/api/frame/state": {
                "method": "GET",
                "response_fields": [
                    "pending",
                    "pending_submission_ids",
                    "ready",
                    "ready_submission_ids",
                    "canceled_submission_ids",
                ],
                "response_type": "frame_state",
            },
            "/api/frame/output/poll": {
                "method": "GET",
                "query_fields": ["submission_id"],
                "response_fields": ["ready", "status", "output"],
                "response_type": "frame_output_poll",
            },
            "/api/frame/output/ready": {
                "method": "GET",
                "response_fields": ["ready", "outputs"],
                "response_type": "frame_output_ready",
            },
            "/api/frame/output/poll_all": {
                "method": "GET",
                "response_fields": ["outputs"],
                "response_type": "frame_output_poll_all",
            },
            "/api/frame/output/status": {
                "method": "GET",
                "query_fields": ["submission_id"],
                "response_fields": ["submission_id", "status"],
                "response_type": "frame_submission_status",
            },
            "/api/table/materials": {"method": "GET", "response_type": "materials_table"},
            "/api/table/gases": {"method": "GET", "response_type": "gases_table"},
            "/api/table/lights": {"method": "GET", "response_type": "lights_table"},
            "/api/table/optics": {"method": "GET", "response_type": "optics_table"},
            "/api/table/reactions": {"method": "GET", "response_type": "reactions_table"},
            "/api/material/write": {
                "method": "POST",
                "body_fields": ["x", "y", "material", "radius", "immediate", *inline_target_query_optional_fields],
                "request_type": "material_injector",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/material/fill": {
                "method": "POST",
                "body_fields": ["x", "y", "width", "height", "material", "immediate", *inline_target_query_optional_fields],
                "request_type": "material_fill_request",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/inject/temperature": {
                "method": "POST",
                "body_fields": ["x", "y", "delta", "radius", "immediate", *inline_target_query_optional_fields],
                "request_type": "temperature_injector",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/inject/velocity": {
                "method": "POST",
                "body_fields": ["x", "y", "velocity", "radius", "carrier", "mode", "immediate", *inline_target_query_optional_fields],
                "request_type": "velocity_injector",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/inject/gas": {
                "method": "POST",
                "body_fields": ["x", "y", "species", "amount", "radius", "immediate", *inline_target_query_optional_fields],
                "request_type": "gas_injector",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/inject/force": {
                "method": "POST",
                "body_fields": [*force_source_fields, "immediate", *inline_target_query_optional_fields],
                "request_type": "force_injector",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/force_sources/set": {
                "method": "POST",
                "body_fields": ["force_sources"],
                "request_type": "force_source[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "replace_semantics": "replace_all",
                "queueing": "deferred_command",
            },
            "/api/emitters/set": {
                "method": "POST",
                "body_fields": ["emitters"],
                "request_type": "emitter[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "replace_semantics": "replace_all",
                "queueing": "deferred_command",
            },
            "/api/inject/light": {
                "method": "POST",
                "body_fields": [*emitter_fields, "immediate", *inline_target_query_optional_fields],
                "request_type": "light_injector",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/readback/request": {
                "method": "POST",
                "body_fields": [
                    "request_id",
                    "center_x",
                    "center_y",
                    "width",
                    "height",
                    "channels",
                    "observer_id",
                    "label",
                    *inline_target_query_optional_fields,
                ],
                    "request_type": "readback_request",
                    "response_fields": ["ok", "queued", "pending_commands", "request_id"],
                    "response_type": "deferred_readback_request_result",
                    "queueing": "deferred_command",
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
            "/api/readback/plan": {
                "method": "POST",
                "body_fields": [
                    "request_id",
                    "center_x",
                    "center_y",
                    "width",
                    "height",
                    "channels",
                    "observer_id",
                    "label",
                    *inline_target_query_optional_fields,
                ],
                "request_type": "readback_request",
                "response_fields": ["ok", *readback_plan_fields],
                "response_type": "readback_plan_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/readback/preview": {
                "method": "POST",
                "body_fields": [
                    "request_id",
                    "center_x",
                    "center_y",
                    "width",
                    "height",
                    "channels",
                    "observer_id",
                    "label",
                    *inline_target_query_optional_fields,
                ],
                "request_type": "readback_request",
                "response_fields": ["ok", "request"],
                "response_type": "readback_preview_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/frame/preview": {
                "method": "POST",
                "request_type": "frame_input",
                "response_fields": ["ok", "preview"],
                "response_type": "frame_preview_result",
            },
            "/api/frame/request": {
                "method": "POST",
                "request_type": "frame_input",
                "response_fields": ["ok", "queued", "pending_frames", "submission_id", "preview"],
                "response_type": "frame_submit_result",
                "queueing": "deferred_frame",
            },
            "/api/frame/submit": {
                "method": "POST",
                "request_type": "frame_input",
                "response_fields": ["ok", "queued", "pending_frames", "submission_id"],
                "response_type": "deferred_frame_submit_ack",
            },
            "/api/frame/cycle": {
                "method": "POST",
                "body_fields": ["apply_frame"],
                "request_type": "frame_input",
                "response_fields": ["ok", "applied", "queued", "pending_frames", "submission_id", "preview", "result"],
                "response_type": "frame_cycle_result",
                "queueing": "deferred_frame_when_applied",
            },
            "/api/frame/cancel": {
                "method": "POST",
                "body_fields": ["submission_id"],
                "response_fields": ["ok", "submission_id", "status", "pending_frames"],
                "request_type": "frame_cancel_request",
                "response_type": "frame_cancel_result",
            },
            "/api/frame/cancel_all": {
                "method": "POST",
                "response_fields": ["ok", "canceled_submission_ids", "pending_frames"],
                "response_type": "frame_cancel_all_result",
            },
            "/api/entity/states": {
                "method": "GET",
                "response_fields": ["entities"],
                "response_type": "entity_states_response",
            },
            "/api/entity/observations/state": {
                "method": "GET",
                "response_fields": ["observations", "targets", "requests"],
                "response_type": "entity_observation_runtime",
            },
            "/api/entity/observations/consumed": {
                "method": "GET",
                "response_fields": ["frame_id", "consumed", "consumed_readbacks", "observations", "entity_feedback"],
                "response_type": "entity_observation_consume_result",
            },
            "/api/entity/placeholders/state": {
                "method": "GET",
                "response_fields": ["placeholders"],
                "response_type": "entity_placeholders_response",
            },
            "/api/entity/feedback": {
                "method": "GET",
                "response_fields": ["feedback"],
                "response_type": "entity_feedback_response",
            },
            "/api/entity/controller/state": {
                "method": "GET",
                "response_fields": ["controller_state"],
                "response_type": "entity_controller_state",
            },
            "/api/entity/controller/state/set": {
                "method": "POST",
                "body_fields": ["controller_state"],
                "request_type": "entity_controller_state",
                "response_fields": ["ok", "queued", "pending_frames", "submission_id"],
                "response_type": "deferred_controller_state_result",
                "queueing": "deferred_frame",
            },
            "/api/entity/observations/set": {
                "method": "POST",
                "body_fields": ["observations"],
                "request_type": "entity_observation_spec[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "replace_semantics": "replace_all",
                "queueing": "deferred_command",
            },
            "/api/entity/observations/request": {
                "method": "POST",
                "body_fields": ["request_id", *observation_target_fields, "target_queries"],
                "request_type": "observation_target",
                "response_fields": ["ok", "queued", "pending_commands", "request_id"],
                "response_type": "deferred_observation_request_result",
                "queueing": "deferred_command",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/entity/observations/preview": {
                "method": "POST",
                "body_fields": ["request_id", *observation_target_fields, "target_queries"],
                "request_type": "observation_target",
                "response_fields": ["ok", "request"],
                "response_type": "readback_preview_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/entity/observations/plan": {
                "method": "POST",
                "body_fields": ["request_id", *observation_target_fields, "target_queries"],
                "request_type": "observation_target",
                "response_fields": ["ok", *observation_plan_fields],
                "response_type": "observation_plan_result",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/entity/observations/consume": {
                "method": "POST",
                "response_fields": ["frame_id", "consumed", "consumed_readbacks", "observations", "entity_feedback"],
                "response_type": "entity_observation_consume_result",
            },
            "/api/entity/placeholders": {
                "method": "POST",
                "body_fields": ["placeholders"],
                "request_type": "entity_placeholder[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "replace_semantics": "replace_all",
                "queueing": "deferred_command",
            },
            "/api/entity/states/set": {
                "method": "POST",
                "body_fields": ["entities"],
                "request_type": "entity_state[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "replace_semantics": "replace_all",
                "queueing": "deferred_command",
            },
            "/api/entity/states/patch": {
                "method": "POST",
                "body_fields": ["patches"],
                "request_type": "entity_state_patch[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/entity/controller/turn": {
                "method": "POST",
                "body_fields": [
                    "controller_state",
                    "focus_center",
                    "entities",
                    "entity_placeholders",
                    "patches",
                    "observation_specs",
                    "force_sources",
                    "emitters",
                    "target_queries",
                    "change_intents",
                    "carrier_intents",
                    "observation_targets",
                    "readback_requests",
                    "commands",
                ],
                "request_type": "entity_controller_turn",
                "response_fields": ["ok", "queued", "pending_frames", "submission_id", "preview"],
                "response_type": "entity_controller_submit_result",
                "queueing": "deferred_frame",
            },
            "/api/entity/controller/preview": {
                "method": "POST",
                "body_fields": [
                    "controller_state",
                    "focus_center",
                    "entities",
                    "entity_placeholders",
                    "patches",
                    "observation_specs",
                    "force_sources",
                    "emitters",
                    "target_queries",
                    "change_intents",
                    "carrier_intents",
                    "observation_targets",
                    "readback_requests",
                    "commands",
                ],
                "request_type": "entity_controller_turn",
                "response_fields": ["ok", "preview"],
                "response_type": "entity_controller_preview_result",
            },
            "/api/entity/controller/submit": {
                "method": "POST",
                "body_fields": [
                    "controller_state",
                    "focus_center",
                    "entities",
                    "entity_placeholders",
                    "patches",
                    "observation_specs",
                    "force_sources",
                    "emitters",
                    "target_queries",
                    "change_intents",
                    "carrier_intents",
                    "observation_targets",
                    "readback_requests",
                    "commands",
                ],
                "request_type": "entity_controller_turn",
                "response_fields": ["ok", "queued", "pending_frames", "submission_id", "preview"],
                "response_type": "entity_controller_submit_result",
                "queueing": "deferred_frame",
            },
            "/api/entity/controller/cycle": {
                "method": "POST",
                "body_fields": [
                    "apply_turn",
                    "controller_state",
                    "focus_center",
                    "entities",
                    "entity_placeholders",
                    "patches",
                    "observation_specs",
                    "force_sources",
                    "emitters",
                    "target_queries",
                    "change_intents",
                    "carrier_intents",
                    "observation_targets",
                    "readback_requests",
                    "commands",
                ],
                "request_type": "entity_controller_cycle",
                "response_fields": ["ok", "applied", "queued", "pending_frames", "submission_id", "preview", "result"],
                "response_type": "entity_controller_cycle_result",
                "queueing": "deferred_frame_when_applied",
            },
            "/api/paging/focus": {
                "method": "POST",
                "body_fields": ["x", "y", *inline_target_query_optional_fields],
                "response_fields": ["ok", "queued", "pending_commands", "target_center"],
                "request_type": "paging_focus_request",
                "response_type": "paging_focus_result",
                "queueing": "deferred_command",
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
            },
            "/api/paging/store/state": {
                "method": "GET",
                "response_fields": ["stored_stripes", "key_listing_supported", "stripe_keys"],
                "response_type": "page_store_state",
            },
            "/api/paging/store/has": {
                "method": "POST",
                "body_fields": ["update"],
                "request_type": "page_stripe_update",
                "response_fields": ["stored"],
                "response_type": "page_store_has_result",
            },
            "/api/paging/store/capture": {
                "method": "POST",
                "body_fields": ["update"],
                "request_type": "page_stripe_update",
                "response_fields": ["ok", "stored_stripes", "payload"],
                "response_type": "page_store_capture_result",
            },
            "/api/paging/store/load": {
                "method": "POST",
                "body_fields": ["update"],
                "request_type": "page_stripe_update",
                "response_fields": ["ok", "stored", "payload"],
                "response_type": "page_store_load_result",
            },
            "/api/paging/store/apply": {
                "method": "POST",
                "body_fields": ["update", "immediate"],
                "request_type": "page_stripe_update",
                "response_fields": ["ok", "stored", "queued", "pending_commands"],
                "response_type": "page_store_apply_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
            },
            "/api/paging/store/save": {
                "method": "POST",
                "body_fields": ["update", "payload"],
                "request_type": "page_stripe_apply",
                "response_fields": ["ok", "stored_stripes"],
                "response_type": "page_store_save_result",
            },
            "/api/paging/store/export": {
                "method": "GET",
                "response_fields": ["stored_stripes", "key_listing_supported", "entries"],
                "response_type": "page_store_export",
            },
            "/api/paging/store/import": {
                "method": "POST",
                "body_fields": ["entries", "clear"],
                "request_type": "page_store_import",
                "response_fields": ["ok", "cleared", "imported", "stored_stripes"],
                "response_type": "page_store_import_result",
            },
            "/api/paging/store/clear": {
                "method": "POST",
                "response_fields": ["ok", "cleared", "stored_stripes"],
                "response_type": "page_store_clear_result",
            },
            "/api/paging/stripe/capture": {
                "method": "POST",
                "body_fields": ["update"],
                "request_type": "page_stripe_update",
                "response_fields": ["ok", "payload"],
                "response_type": "page_stripe_capture_result",
            },
            "/api/paging/stripe/apply": {
                "method": "POST",
                "body_fields": ["update", "payload", "immediate"],
                "request_type": "page_stripe_apply",
                "response_fields": ["ok", "queued", "pending_commands"],
                "response_type": "page_stripe_apply_result",
                "queueing": "deferred_command",
                "immediate_supported": True,
            },
            "/api/table/material": {
                "method": "POST",
                "body_fields": ["name", "fields"],
                "request_type": "material_patch",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/light": {
                "method": "POST",
                "body_fields": ["name", "fields"],
                "request_type": "light_patch",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/gas": {
                "method": "POST",
                "body_fields": ["name", "fields"],
                "request_type": "gas_patch",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/optic": {
                "method": "POST",
                "body_fields": ["material_name", "light_type", "fields"],
                "request_type": "optics_patch",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/reaction": {
                "method": "POST",
                "body_fields": ["index", "fields"],
                "request_type": "reaction_action_patch",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/reaction/delete": {
                "method": "POST",
                "body_fields": ["index"],
                "request_type": "reaction_action_delete_request",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/reaction_rule": {
                "method": "POST",
                "body_fields": ["rule_set", "index", "fields"],
                "request_type": "reaction_rule_patch",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/reaction_rule/delete": {
                "method": "POST",
                "body_fields": ["rule_set", "index"],
                "request_type": "reaction_rule_delete_request",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/materials/update": {
                "method": "POST",
                "body_fields": ["materials"],
                "request_type": "material[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "update_semantics": "merge_by_material_id",
                "queueing": "deferred_command",
            },
            "/api/table/gases/update": {
                "method": "POST",
                "body_fields": ["gases"],
                "request_type": "gas[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "update_semantics": "merge_by_species_id",
                "queueing": "deferred_command",
            },
            "/api/table/lights/update": {
                "method": "POST",
                "body_fields": ["lights"],
                "request_type": "light[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "update_semantics": "merge_by_light_type_id",
                "queueing": "deferred_command",
            },
            "/api/table/optics/update": {
                "method": "POST",
                "body_fields": ["optics"],
                "request_type": "optics[]",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "update_semantics": "merge_by_material_name_and_light_type",
                "queueing": "deferred_command",
            },
            "/api/table/reactions/update": {
                "method": "POST",
                "body_fields": ["actions", "rules"],
                "request_type": "reaction_table_append",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "queueing": "deferred_command",
            },
            "/api/table/reactions/replace": {
                "method": "POST",
                "body_fields": ["actions", "rules"],
                "request_type": "reaction_table_replace",
                "response_fields": queued_mutation_result_fields,
                "response_type": "queued_mutation_result",
                "replace_semantics": "replace_all",
                "queueing": "deferred_command",
            },
            "/api/control/pause": {"method": "POST", "response_fields": ["paused"], "response_type": "pause_result"},
            "/api/control/resume": {"method": "POST", "response_fields": ["paused"], "response_type": "resume_result"},
            "/api/control/step": {"method": "POST", "response_fields": ["single_step"], "response_type": "step_result"},
            "/api/control/speed": {
                "method": "POST",
                "body_fields": ["speed"],
                "response_fields": ["speed"],
                "request_type": "speed_request",
                "response_type": "speed_result",
            },
            "/api/control/reset": {
                "method": "POST",
                "response_fields": ["ok", "queued", "pending_commands"],
                "response_type": "control_reset_result",
                "queueing": "deferred_command",
            },
        }
        return {
            "world": {
                "buffer_size": [int(self.width), int(self.height)],
                "active_size": [int(self.paging.active_width), int(self.paging.active_height)],
                "gas_cell_size": int(self.gas_cell_size),
                "gas_grid_size": [int(self.gas_width), int(self.gas_height)],
                "tile_size": int(self.active.tile_size),
                "tile_grid_size": [int(self.active.tile_width), int(self.active.tile_height)],
                "chunk_tiles": int(self.active.chunk_tiles),
                "chunk_grid_size": [int(self.active.chunk_width), int(self.active.chunk_height)],
                "default_debug_view": self.default_debug_view.value,
            },
            "material_name_aliases": {
                "accepts_runtime_names": True,
                "accepts_base_aliases": True,
                "base_to_runtime": material_name_aliases,
                "fields": {
                    "material": ["material"],
                    "material_def": [
                        "name",
                        "collapse_generation",
                        "powder_generation",
                        "melt_to_material",
                        "freeze_to_material",
                    ],
                    "gas_def": ["condense_to_material"],
                    "material_optics": ["material_name"],
                    "reaction_action": ["target_material", "emit_material"],
                    "pair_reaction_rule": ["lhs_material", "rhs_material"],
                    "self_reaction_rule": ["material"],
                    "entity_state": ["placeholder_material"],
                    "entity_placeholder": ["material"],
                },
            },
            "debug_views": [view.value for view in DebugView],
            "debug_view_options": deepcopy(debug_view_options),
            "http_api": {
                "base_path": "/api",
                "transport": "json_over_http",
                "endpoints": http_api_endpoints,
            },
            "readback_channels": list(READBACK_ALLOWED_CHANNELS),
            "readback": {
                "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                "max_async_window_size": [int(MAX_ASYNC_READBACK_WIDTH), int(MAX_ASYNC_READBACK_HEIGHT)],
                "request_fields": readback_request_fields,
                "result_fields": ["frame_id", "request", "payload"],
                "channel_payload_types": dict(readback_channel_payload_types),
            },
            "target_query": {
                "terrain_anchor_filters": sorted(TERRAIN_ANCHOR_FILTERS),
                "ignored_anchor_filters": sorted(IGNORED_ANCHOR_FILTERS),
                "directions": [*CARDINAL_DIRECTION_VECTORS.keys(), "forward", "backward"],
                "distance_hint_cells": {
                    hint: int(distance)
                    for hint, distance in TARGET_QUERY_DISTANCE_HINT_CELLS.items()
                },
                "cells_per_meter": float(TARGET_QUERY_CELLS_PER_METER),
            },
            "phases": {
                phase.name.lower(): int(phase)
                for phase in Phase
            },
            "reaction_types": [reaction_type.name.lower() for reaction_type in ReactionType],
            "collapse_behaviors": sorted(COLLAPSE_BEHAVIOR_IDS),
            "tag_bits": deepcopy(self.tag_bits_by_name),
            "injectors": {
                "material": {
                    "fields": ["x", "y", "material", "radius"],
                    "material_aliases": material_name_aliases,
                    "supports_immediate": True,
                    "optional_fields": ["immediate", *inline_target_query_optional_fields],
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
                "temperature": {
                    "fields": ["x", "y", "delta", "radius"],
                    "supports_immediate": True,
                    "optional_fields": ["immediate", *inline_target_query_optional_fields],
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
                "velocity": {
                    "fields": ["x", "y", "velocity", "radius", "carrier", "mode"],
                    "carriers": ["cell", "flow", "both"],
                    "modes": ["add", "set"],
                    "supports_immediate": True,
                    "optional_fields": ["immediate", *inline_target_query_optional_fields],
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
                "gas": {
                    "fields": ["x", "y", "species", "amount", "radius"],
                    "supports_immediate": True,
                    "optional_fields": ["immediate", *inline_target_query_optional_fields],
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
                "force": {
                    "fields": ["x", "y", "direction", "radius", "strength", "lifetime"],
                    "supports_immediate": True,
                    "optional_fields": ["immediate", *inline_target_query_optional_fields],
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
                "light": {
                    "fields": emitter_fields,
                    "supports_direction": True,
                    "supports_spread": True,
                    "default_spread": 0.25,
                    "supports_immediate": True,
                    "optional_fields": ["immediate", *inline_target_query_optional_fields],
                    "target_query_fields": target_query_fields,
                    "supports_inline_target_queries": True,
                },
            },
            "force_sources": {
                "fields": force_source_fields,
                "replace_semantics": "replace_all",
            },
            "entity_state": {
                "fields": entity_state_fields,
                "replace_semantics": "replace_all",
                "material_alias_fields": ["placeholder_material"],
                "field_types": {
                    "entity_id": {"type": "int"},
                    "x": {"type": "int"},
                    "y": {"type": "int"},
                    "width": {"type": "int"},
                    "height": {"type": "int"},
                    "velocity_xy": {"type": "float2"},
                    "facing_xy": {"type": "float2", "optional": True},
                    "placeholder_material": {"type": "str"},
                    "tags": {"type": "str[]"},
                    "observe_channels": {"type": "str[]"},
                    "observe_pad_cells": {"type": "int"},
                    "observe_width": {"type": "int", "optional": True},
                    "observe_height": {"type": "int", "optional": True},
                    "observe_label": {"type": "str", "optional": True},
                },
            },
            "entity_placeholder": {
                "fields": entity_placeholder_fields,
                "replace_semantics": "replace_all",
                "material_alias_fields": ["material"],
                "field_types": {
                    "entity_id": {"type": "int"},
                    "x": {"type": "int"},
                    "y": {"type": "int"},
                    "width": {"type": "int"},
                    "height": {"type": "int"},
                    "material": {"type": "str"},
                },
            },
            "entity_placeholder_runtime": {
                "fields": ["entity_id", "bbox", "cells"],
                "cell_fields": [
                    "x",
                    "y",
                    "material_id",
                    "material",
                    "phase",
                    "displaced_material_id",
                    "displaced_material",
                ],
            },
            "entity_feedback": {
                "fields": ["entity_id", "bbox", "cells"],
                "cell_fields": ["x", "y", "present", "material_id", "phase", "integrity", "entity_id"],
            },
            "entity_controller_state": {
                "fields": ["controller_state"],
                "controller_state_type": "json",
                "replace_semantics": "replace_all",
                "persistence": "cpu_side",
            },
            "entity_observation_spec": {
                "fields": entity_observation_spec_fields,
                "replace_semantics": "replace_all",
            },
            "entity_state_patch": {
                "fields": entity_state_patch_fields,
                "patchable_fields": sorted(ENTITY_STATE_PATCHABLE_FIELDS),
                "material_alias_fields": ["placeholder_material"],
                "field_types": {
                    "entity_id": {"type": "int"},
                    "fields": {"type": "json"},
                },
            },
            "entity_controller_turn": {
                "fields": [
                    "controller_state",
                    "focus_center",
                    "entities",
                    "entity_placeholders",
                    "patches",
                    "observation_specs",
                    "force_sources",
                    "emitters",
                    "target_queries",
                    "change_intents",
                    "carrier_intents",
                    "observation_targets",
                    "readback_requests",
                    "commands",
                ],
            },
            "entity_controller_cycle": {
                "fields": [
                    "apply_turn",
                    "controller_state",
                    "focus_center",
                    "entities",
                    "entity_placeholders",
                    "patches",
                    "observation_specs",
                    "force_sources",
                    "emitters",
                    "target_queries",
                    "change_intents",
                    "carrier_intents",
                    "observation_targets",
                    "readback_requests",
                    "commands",
                ],
            },
            "resolved_target": {
                "field_order": resolved_target_fields,
                "fields": {
                    "query_id": {"type": "str"},
                    "status": {"type": "str"},
                    "anchor_filters": {"type": "str[]"},
                    "direction": {"type": "str", "optional": True},
                    "distance_cells": {"type": "int"},
                    "distance_meters": {"type": "float", "optional": True},
                    "distance_hint": {"type": "str", "optional": True},
                    "label": {"type": "str", "optional": True},
                    "source_position": {"type": "cell_xy", "optional": True},
                    "source_world_position": {"type": "cell_xy", "optional": True},
                    "anchor_kind": {"type": "str", "optional": True},
                    "anchor_entity_id": {"type": "int", "optional": True},
                    "anchor_position": {"type": "cell_xy", "optional": True},
                    "anchor_world_position": {"type": "cell_xy", "optional": True},
                    "resolved_position": {"type": "cell_xy", "optional": True},
                    "resolved_world_position": {"type": "cell_xy", "optional": True},
                    "note": {"type": "str", "optional": True},
                },
            },
            "resolved_change_intent": {
                "field_order": resolved_change_intent_fields,
                "fields": {
                    "intent_id": {"type": "str"},
                    "status": {"type": "str"},
                    "target_query_id": {"type": "str", "optional": True},
                    "label": {"type": "str", "optional": True},
                    "potency": {"type": "float"},
                    "stability": {"type": "float"},
                    "center_position": {"type": "cell_xy", "optional": True},
                    "center_world_position": {"type": "cell_xy", "optional": True},
                    "effective_radius": {"type": "int"},
                    "material": {"type": "str", "optional": True},
                    "temperature_delta": {"type": "float"},
                    "velocity": {"type": "float2", "optional": True},
                    "velocity_carrier": {"type": "str"},
                    "velocity_mode": {"type": "str"},
                    "require_empty": {"type": "bool"},
                    "fallback_mode": {"type": "str"},
                    "fallback_applied": {"type": "bool"},
                    "effect_shape": {"type": "str"},
                    "effect_cells": {"type": "cell_xy[]"},
                    "effect_bounds": {"type": "cell_rect", "optional": True},
                    "generated_commands": {"type": "world_command[]"},
                    "note": {"type": "str", "optional": True},
                },
            },
            "resolved_carrier_intent": {
                "field_order": resolved_carrier_intent_fields,
                "fields": {
                    "intent_id": {"type": "str"},
                    "status": {"type": "str"},
                    "kind": {"type": "str"},
                    "target_query_id": {"type": "str", "optional": True},
                    "label": {"type": "str", "optional": True},
                    "release_mode": {"type": "str"},
                    "potency": {"type": "float"},
                    "stability": {"type": "float"},
                    "source_position": {"type": "cell_xy", "optional": True},
                    "source_world_position": {"type": "cell_xy", "optional": True},
                    "impact_position": {"type": "cell_xy", "optional": True},
                    "impact_world_position": {"type": "cell_xy", "optional": True},
                    "effective_radius": {"type": "int"},
                    "material": {"type": "str", "optional": True},
                    "gas_species": {"type": "str", "optional": True},
                    "gas_amount": {"type": "float"},
                    "light_type": {"type": "str", "optional": True},
                    "light_strength": {"type": "float"},
                    "light_spread": {"type": "float"},
                    "force_radius": {"type": "float"},
                    "force_strength": {"type": "float"},
                    "force_lifetime": {"type": "float"},
                    "direction": {"type": "float2", "optional": True},
                    "require_empty": {"type": "bool"},
                    "fallback_mode": {"type": "str"},
                    "fallback_applied": {"type": "bool"},
                    "effect_shape": {"type": "str"},
                    "effect_cells": {"type": "cell_xy[]"},
                    "effect_bounds": {"type": "cell_rect", "optional": True},
                    "generated_commands": {"type": "world_command[]"},
                    "note": {"type": "str", "optional": True},
                },
            },
            "observation_result": {
                "fields": observation_result_fields,
                "request_type": "readback_request",
                "payload_type": "json",
                "payload_schema_type": "readback_payload",
            },
            "entity_observation_runtime": {
                "fields": entity_observation_runtime_fields,
                "field_types": {
                    "observations": {"type": "entity_observation_spec[]"},
                    "targets": {"type": "observation_target[]"},
                    "requests": {"type": "readback_request[]"},
                },
            },
            "entity_observation_consume_result": {
                "fields": entity_observation_consume_result_fields,
                "field_types": {
                    "frame_id": {"type": "int"},
                    "consumed": {"type": "int"},
                    "consumed_readbacks": {"type": "readback_result[]"},
                    "observations": {"type": "observation_result{}", "key": "observer_id"},
                    "entity_feedback": {"type": "entity_feedback{}", "key": "entity_id"},
                },
            },
            "entity_controller_turn_result": {
                "fields": [
                    "frame_id",
                    "controller_state",
                    "consumed",
                    "paging_updates",
                    "resolved_targets",
                    "resolved_change_intents",
                    "resolved_carrier_intents",
                    "resolved_commands",
                    "observation_requests",
                    "readback_requests",
                    "queued_observations",
                    "queued_readbacks",
                    "queued_commands",
                    "entities",
                    "placeholders",
                    "observation_state",
                    "paging_state",
                    "readback_state",
                    "force_sources",
                    "emitters",
                    "pending_commands",
                ],
                "field_types": {
                    "frame_id": {"type": "int"},
                    "controller_state": {"type": "json"},
                    "consumed": {"type": "entity_observation_consume_result"},
                    "paging_updates": {"type": "page_stripe_update[]"},
                    "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                    "resolved_change_intents": {"type": "resolved_change_intent{}", "key": "intent_id"},
                    "resolved_carrier_intents": {"type": "resolved_carrier_intent{}", "key": "intent_id"},
                    "resolved_commands": {"type": "world_command[]"},
                    "observation_requests": {"type": "readback_request[]"},
                    "readback_requests": {"type": "readback_request[]"},
                    "queued_observations": {"type": "int"},
                    "queued_readbacks": {"type": "int"},
                    "queued_commands": {"type": "int"},
                    "entities": {"type": "entity_state[]"},
                    "placeholders": {"type": "entity_placeholder_runtime[]"},
                    "observation_state": {"type": "entity_observation_runtime"},
                    "paging_state": {"type": "paging_state"},
                    "readback_state": {"type": "readback_state"},
                    "force_sources": {"type": "force_source[]"},
                    "emitters": {"type": "emitters"},
                    "pending_commands": {"type": "pending_commands"},
                },
            },
            "entity_controller_turn_preview": {
                "fields": [
                    "frame_id",
                    "controller_state",
                    "consumed",
                    "paging_updates",
                    "resolved_targets",
                    "resolved_change_intents",
                    "resolved_carrier_intents",
                    "resolved_commands",
                    "observation_requests",
                    "observation_plans",
                    "readback_requests",
                    "readback_plans",
                    "bridge_frame_snapshot",
                    "queued_observations",
                    "queued_readbacks",
                    "queued_commands",
                    "placeholder_count",
                    "entities",
                    "placeholders",
                    "observation_state",
                    "paging_state",
                    "force_sources",
                    "emitters",
                    "pending_commands",
                ],
                "field_types": {
                    "frame_id": {"type": "int"},
                    "controller_state": {"type": "json"},
                    "consumed": {"type": "entity_observation_consume_result"},
                    "paging_updates": {"type": "page_stripe_update[]"},
                    "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                    "resolved_change_intents": {"type": "resolved_change_intent{}", "key": "intent_id"},
                    "resolved_carrier_intents": {"type": "resolved_carrier_intent{}", "key": "intent_id"},
                    "resolved_commands": {"type": "world_command[]"},
                    "observation_requests": {"type": "readback_request[]"},
                    "observation_plans": {"type": "observation_plan[]"},
                    "readback_requests": {"type": "readback_request[]"},
                    "readback_plans": {"type": "readback_plan[]"},
                    "bridge_frame_snapshot": {"type": "bridge_frame_snapshot"},
                    "queued_observations": {"type": "int"},
                    "queued_readbacks": {"type": "int"},
                    "queued_commands": {"type": "int"},
                    "placeholder_count": {"type": "int"},
                    "entities": {"type": "entity_state[]"},
                    "placeholders": {"type": "entity_placeholder_runtime[]"},
                    "observation_state": {"type": "entity_observation_runtime"},
                    "paging_state": {"type": "paging_state"},
                    "force_sources": {"type": "force_source[]"},
                    "emitters": {"type": "emitters"},
                    "pending_commands": {"type": "pending_commands"},
                },
            },
            "entity_controller_cycle_result": {
                "fields": ["applied", "queued", "pending_frames", "submission_id", "preview", "result"],
                "preview_type": "entity_controller_turn_preview",
                "result_type": "entity_controller_turn_result",
                "result_optional_when_unapplied": True,
                "result_optional_when_deferred": True,
                "submission_id_optional_when_unapplied": True,
                "queueing": "deferred_frame_when_applied",
            },
            "entity_controller_submit_result": {
                "fields": ["ok", "queued", "pending_frames", "submission_id", "preview"],
                "preview_type": "entity_controller_turn_preview",
                "submission_id_optional": False,
                "queueing": "deferred_frame",
            },
            "entity_states_response": {
                "fields": ["entities"],
                "field_types": {
                    "entities": {"type": "entity_state[]"},
                },
            },
            "entity_placeholders_response": {
                "fields": ["placeholders"],
                "field_types": {
                    "placeholders": {"type": "entity_placeholder_runtime[]"},
                },
            },
            "entity_feedback_response": {
                "fields": ["feedback"],
                "field_types": {
                    "feedback": {"type": "entity_feedback{}", "key": "entity_id"},
                },
            },
            "target_preview_result": {
                "fields": ["ok", "resolved_targets"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                },
            },
            "command_preview_result": {
                "fields": ["ok", "command"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "command": {"type": "world_command"},
                },
            },
            "command_request_result": {
                "fields": ["ok", "queued", "pending_commands", "command"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                    "command": {"type": "world_command"},
                },
                "queueing": "deferred_command",
            },
            "change_intent_preview_result": {
                "fields": ["ok", "resolved_intent"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "resolved_intent": {"type": "resolved_change_intent"},
                },
            },
            "change_intent_request_result": {
                "fields": ["ok", "queued", "pending_commands", "resolved_intent"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                    "resolved_intent": {"type": "resolved_change_intent"},
                },
                "queueing": "deferred_command",
            },
            "carrier_intent_preview_result": {
                "fields": ["ok", "resolved_intent"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "resolved_intent": {"type": "resolved_carrier_intent"},
                },
            },
            "carrier_intent_request_result": {
                "fields": ["ok", "queued", "pending_commands", "resolved_intent"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                    "resolved_intent": {"type": "resolved_carrier_intent"},
                },
                "queueing": "deferred_command",
            },
            "readback_preview_result": {
                "fields": ["ok", "request"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "request": {"type": "readback_request"},
                },
            },
            "readback_plan_result": {
                "fields": ["ok", *readback_plan_fields],
                "field_types": {
                    "ok": {"type": "bool"},
                    "request": {"type": "readback_request"},
                    "layout": {"type": "json"},
                    "nbytes": {"type": "int"},
                    "gpu_source_count": {"type": "int"},
                    "cpu_chunk_count": {"type": "int"},
                    "payload": {"type": "json"},
                },
            },
            "observation_plan_result": {
                "fields": ["ok", *observation_plan_fields],
                "field_types": {
                    "ok": {"type": "bool"},
                    "target": {"type": "observation_target"},
                    "request": {"type": "readback_request"},
                    "layout": {"type": "json"},
                    "nbytes": {"type": "int"},
                    "gpu_source_count": {"type": "int"},
                    "cpu_chunk_count": {"type": "int"},
                    "payload": {"type": "json"},
                },
            },
            "frame_preview_result": {
                "fields": ["ok", "preview"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "preview": {"type": "frame_preview"},
                },
            },
            "entity_controller_preview_result": {
                "fields": ["ok", "preview"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "preview": {"type": "entity_controller_turn_preview"},
                },
            },
            "persistent_emitters": {
                "fields": emitter_fields,
                "replace_semantics": "replace_all",
            },
            "emitters": {
                "fields": emitter_runtime_fields,
                "field_types": {
                    "persistent_emitters": {"type": "emitter[]"},
                    "queued_emitters": {"type": "emitter[]"},
                },
            },
            "force_sources_response": {
                "fields": ["force_sources"],
                "field_types": {
                    "force_sources": {"type": "force_source[]"},
                },
            },
            "cell_window": {
                "fields": cell_window_fields,
                "origin_type": "cell_xy",
                "size_type": "cell_wh",
                "temperature_alias": "temperature",
            },
            "temperature_window": {
                "fields": temperature_window_fields,
                "grid": "cell",
            },
            "gas_window": {
                "fields": gas_window_fields,
                "grid": "gas",
                "species_key": "species",
                "concentration_key": "concentration",
            },
            "pressure_window": {
                "fields": pressure_window_fields,
                "grid": "gas",
            },
            "velocity_window": {
                "fields": velocity_window_fields,
                "grid": "gas",
                "vector_components": 2,
            },
            "optics_window": {
                "fields": optics_window_fields,
                "origin_type": "cell_xy",
                "size_type": "cell_wh",
                "gas_origin_type": "gas_xy",
                "gas_size_type": "gas_wh",
                "dose_key_type": "light_name",
            },
            "visible_illumination": {
                "fields": visible_illumination_fields,
                "grid": "cell",
                "color_components": 3,
            },
            "debug_frame": {
                "fields": debug_frame_fields,
                "origin_type": "cell_xy",
                "size_type": "cell_wh",
                "frame_type": "rgb_grid",
                "color_components": 3,
            },
            "bridge_runtime": {
                "fields": bridge_runtime_fields,
                "field_types": {
                    "frame_id": {"type": "int"},
                    "bridge": {"type": "json"},
                    "pending_readbacks": {"type": "int"},
                    "inflight_readbacks": {"type": "int"},
                    "ready_readbacks": {"type": "int"},
                    "pending_commands": {"type": "int"},
                },
            },
            "bridge_resource_catalog": {
                "fields": bridge_resource_catalog_fields,
                "field_types": {
                    "typed_tables": {"type": "json"},
                    "shadow_buffers": {"type": "json"},
                    "snapshots": {"type": "json"},
                },
            },
            "bridge_typed_table_snapshot": {
                "fields": bridge_typed_table_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "rows": {"type": "json"},
                },
            },
            "bridge_typed_table_slice_snapshot": {
                "fields": bridge_typed_table_slice_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "offset": {"type": "int"},
                    "limit": {"type": "int"},
                    "returned_count": {"type": "int"},
                    "slice_shape": {"type": "int[]"},
                    "rows": {"type": "json"},
                },
            },
            "bridge_shadow_buffer_snapshot": {
                "fields": bridge_shadow_buffer_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "rows": {"type": "json"},
                    "values": {"type": "json"},
                    "utf8": {"type": "string", "optional": True},
                },
            },
            "bridge_shadow_buffer_slice_snapshot": {
                "fields": bridge_shadow_buffer_slice_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "offset": {"type": "int"},
                    "limit": {"type": "int"},
                    "returned_count": {"type": "int"},
                    "slice_shape": {"type": "int[]"},
                    "rows": {"type": "json"},
                    "values": {"type": "json"},
                    "utf8": {"type": "string", "optional": True},
                },
            },
            "bridge_shadow_buffer_window_snapshot": {
                "fields": bridge_shadow_buffer_window_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "window_origin": {"type": "int[]"},
                    "requested_size": {"type": "int[]"},
                    "window_size": {"type": "int[]"},
                    "window_axes": {"type": "int[]"},
                    "returned_shape": {"type": "int[]"},
                    "rows": {"type": "json"},
                    "values": {"type": "json"},
                    "utf8": {"type": "string", "optional": True},
                },
            },
            "bridge_shadow_buffer_world_window_snapshot": {
                "fields": bridge_shadow_buffer_spatial_window_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "coord_space": {"type": "string"},
                    "window_origin": {"type": "int[]"},
                    "requested_size": {"type": "int[]"},
                    "window_size": {"type": "int[]"},
                    "window_axes": {"type": "int[]"},
                    "returned_shape": {"type": "int[]"},
                    "rows": {"type": "json"},
                    "values": {"type": "json"},
                    "utf8": {"type": "string", "optional": True},
                },
            },
            "bridge_shadow_buffer_gas_window_snapshot": {
                "fields": bridge_shadow_buffer_spatial_window_fields,
                "field_types": {
                    "name": {"type": "string"},
                    "shape": {"type": "int[]"},
                    "dtype": {"type": "string"},
                    "structured": {"type": "bool"},
                    "field_names": {"type": "string[]"},
                    "row_count": {"type": "int"},
                    "coord_space": {"type": "string"},
                    "window_origin": {"type": "int[]"},
                    "requested_size": {"type": "int[]"},
                    "window_size": {"type": "int[]"},
                    "window_axes": {"type": "int[]"},
                    "returned_shape": {"type": "int[]"},
                    "rows": {"type": "json"},
                    "values": {"type": "json"},
                    "utf8": {"type": "string", "optional": True},
                },
            },
            "bridge_upload_snapshot": {
                "fields": bridge_upload_snapshot_fields,
                "field_types": {
                    "frame_meta": {"type": "json"},
                    "world_commands": {"type": "world_command[]"},
                    "readback_requests": {"type": "json"},
                    "page_stripes": {"type": "json"},
                },
            },
            "bridge_frame_snapshot": {
                "fields": bridge_frame_snapshot_fields,
                "readback_stage_type": "bridge_readback_stage",
                "index_stage_type": "bridge_index_stage",
                "field_types": {
                    "prepared": {"type": "bool"},
                    "commands": {"type": "world_command[]"},
                    "command_stages": {"type": "bridge_index_stage[]"},
                    "readback_requests": {"type": "readback_request[]"},
                    "readback_request_stages": {"type": "bridge_readback_stage[]"},
                    "placeholders": {"type": "entity_placeholder_runtime[]"},
                    "placeholder_stages": {"type": "bridge_index_stage[]"},
                    "placeholder_dirty_rects": {"type": "json"},
                    "paging_updates": {"type": "page_stripe_update[]"},
                    "paging_update_stages": {"type": "bridge_index_stage[]"},
                    "page_stripes": {"type": "json"},
                    "page_stripe_stages": {"type": "bridge_index_stage[]"},
                },
            },
            "bridge_readback_stage": {
                "fields": bridge_readback_stage_fields,
            },
            "bridge_index_stage": {
                "fields": bridge_index_stage_fields,
            },
            "readback_state": {
                "fields": readback_state_fields,
                "request_type": "readback_request",
                "queued_command_type": "world_command",
            },
            "readback_result": {
                "fields": readback_result_fields,
                "request_type": "readback_request",
                "payload_type": "json",
                "payload_schema_type": "readback_payload",
            },
            "readback_poll": {
                "fields": readback_poll_fields,
                "result_type": "readback_result",
                "result_optional_when_not_ready": True,
            },
            "readback_ready": {
                "fields": ["ready", "results"],
                "field_types": {
                    "ready": {"type": "int"},
                    "results": {"type": "readback_result[]"},
                },
            },
            "readback_poll_all": {
                "fields": ["results"],
                "field_types": {
                    "results": {"type": "readback_result[]"},
                },
            },
            "readback_plan": {
                "fields": readback_plan_fields,
                "request_type": "readback_request",
                "layout_type": "json",
                "payload_type": "json",
                "field_types": {
                    "request": {"type": "readback_request"},
                    "layout": {"type": "json"},
                    "nbytes": {"type": "int"},
                    "gpu_source_count": {"type": "int"},
                    "cpu_chunk_count": {"type": "int"},
                    "payload": {"type": "json"},
                },
            },
            "observation_plan": {
                "fields": observation_plan_fields,
                "target_type": "observation_target",
                "request_type": "readback_request",
                "layout_type": "json",
                "payload_type": "json",
                "field_types": {
                    "target": {"type": "observation_target"},
                    "request": {"type": "readback_request"},
                    "layout": {"type": "json"},
                    "nbytes": {"type": "int"},
                    "gpu_source_count": {"type": "int"},
                    "cpu_chunk_count": {"type": "int"},
                    "payload": {"type": "json"},
                },
            },
            "readback_payload": {
                "fields": readback_payload_fields,
                "channel_types": dict(readback_channel_payload_types),
                "optional_fields": list(READBACK_ALLOWED_CHANNELS),
            },
            "readback_cell_payload": {
                "fields": readback_cell_payload_fields,
                "origin_type": "cell_xy",
                "size_type": "cell_wh",
                "packed_core_words": True,
                "core_words_layout_type": "cell_core_layout",
            },
            "readback_scalar_window": {
                "fields": readback_scalar_window_fields,
                "origin_type": "gas_xy",
                "size_type": "gas_wh",
                "grid_key": "grid",
                "values_type": "scalar_grid",
            },
            "readback_vector_window": {
                "fields": readback_vector_window_fields,
                "origin_type": "gas_xy",
                "size_type": "gas_wh",
                "grid_key": "grid",
                "values_type": "vector_grid",
                "vector_components": 2,
            },
            "readback_gas_payload": {
                "fields": readback_gas_payload_fields,
                "origin_type": "gas_xy",
                "size_type": "gas_wh",
                "species_key_type": "gas_name",
            },
            "readback_optics_payload": {
                "fields": readback_optics_payload_fields,
                "origin_type": "cell_xy",
                "size_type": "cell_wh",
                "gas_origin_type": "gas_xy",
                "gas_size_type": "gas_wh",
                "dose_key_type": "light_name",
            },
            "cell_core_layout": {
                "fields": cell_core_layout_fields,
                "word_count": 5,
                "word_bits": 32,
                "packed_words": {
                    "word0": "material_id:u16 | phase:u8 | cell_flags:u8",
                    "word1": "velocity_pack = packHalf2x16(velocity_xy)",
                    "word2": "cell_temperature:f32",
                    "word3": "timer_pack:u32",
                    "word4": "integrity:u16 | reserved0:u16",
                },
                "unpacked_fields": cell_core_unpack_fields,
                "unpack_schema": {
                    "material_id": {"source": "word0", "dtype": "u16"},
                    "phase": {"source": "word0", "dtype": "u8", "bit_range": [16, 23]},
                    "cell_flags": {"source": "word0", "dtype": "u8", "bit_range": [24, 31]},
                    "velocity": {
                        "source": "word1",
                        "encoding": "packHalf2x16",
                        "component_count": 2,
                        "dtype": "f16->f32",
                    },
                    "cell_temperature": {"source": "word2", "dtype": "f32"},
                    "timer_pack": {
                        "source": "word3",
                        "component_count": 4,
                        "component_dtype": "u8",
                    },
                    "integrity": {"source": "word4", "dtype": "u16->f32", "bit_range": [0, 15]},
                },
            },
            "world_command": {
                "fields": ["kind", "payload"],
                "public_kinds": public_world_command_kinds,
                "target_query_overlay_fields": ["target_query_id", "target_dx", "target_dy"],
                "material_alias_fields_by_kind": world_command_material_alias_fields_by_kind,
                "collection_item_types_by_kind": world_command_collection_item_types_by_kind,
                "target_query_supported_kinds": {
                    kind: {"x_field": fields[0], "y_field": fields[1]}
                    for kind, fields in sorted(TARGETED_COMMAND_COORD_FIELDS.items())
                },
                "field_types": {
                    "kind": {"type": "str"},
                    "payload": {"type": "json"},
                },
            },
            "force_source": {
                "fields": force_source_fields,
                "field_types": {
                    "x": {"type": "float"},
                    "y": {"type": "float"},
                    "direction": {"type": "float2"},
                    "radius": {"type": "float"},
                    "strength": {"type": "float"},
                    "lifetime": {"type": "float"},
                },
            },
            "emitter": {
                "fields": emitter_fields,
                "field_types": {
                    "x": {"type": "int"},
                    "y": {"type": "int"},
                    "light_type": {"type": "str"},
                    "strength": {"type": "float"},
                    "radius": {"type": "int"},
                    "direction": {"type": "float2"},
                    "spread": {"type": "float"},
                },
            },
            "target_query": {
                "fields": target_query_fields,
                "directions": [*CARDINAL_DIRECTION_VECTORS.keys(), "forward", "backward"],
                "distance_hints": sorted(TARGET_QUERY_DISTANCE_HINT_CELLS),
                "terrain_anchor_filters": sorted(TERRAIN_ANCHOR_FILTERS),
                "ignored_anchor_filters": sorted(IGNORED_ANCHOR_FILTERS),
                "field_types": {
                    "query_id": {"type": "str"},
                    "anchor_filters": {"type": "str[]"},
                    "source_entity_id": {"type": "int", "optional": True},
                    "source_x": {"type": "int", "optional": True},
                    "source_y": {"type": "int", "optional": True},
                    "anchor_entity_id": {"type": "int", "optional": True},
                    "direction": {"type": "str", "optional": True},
                    "distance_cells": {"type": "int"},
                    "distance_meters": {"type": "float", "optional": True},
                    "distance_hint": {"type": "str", "optional": True},
                    "require_empty": {"type": "bool"},
                    "search_radius": {"type": "int"},
                    "label": {"type": "str", "optional": True},
                },
            },
            "change_intent": {
                "fields": change_intent_fields,
                "velocity_carriers": ["cell", "flow", "both"],
                "velocity_modes": ["add", "set"],
                "fallback_modes": ["nearest_empty", "source"],
                "material_alias_fields": ["material"],
                "field_types": {
                    "intent_id": {"type": "str"},
                    "target_query_id": {"type": "str", "optional": True},
                    "center_x": {"type": "int", "optional": True},
                    "center_y": {"type": "int", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "radius": {"type": "int"},
                    "material": {"type": "str", "optional": True},
                    "temperature_delta": {"type": "float"},
                    "velocity": {"type": "float2", "optional": True},
                    "velocity_carrier": {"type": "str"},
                    "velocity_mode": {"type": "str"},
                    "require_empty": {"type": "bool"},
                    "fallback_mode": {"type": "str"},
                    "fallback_radius": {"type": "int"},
                    "potency": {"type": "float"},
                    "stability": {"type": "float"},
                    "label": {"type": "str", "optional": True},
                },
            },
            "carrier_intent": {
                "fields": carrier_intent_fields,
                "kinds": ["material", "gas", "light", "force"],
                "release_modes": ["impact", "beam", "projectile"],
                "fallback_modes": ["nearest_empty", "source"],
                "material_alias_fields": ["material"],
                "field_types": {
                    "intent_id": {"type": "str"},
                    "kind": {"type": "str"},
                    "target_query_id": {"type": "str", "optional": True},
                    "center_x": {"type": "int", "optional": True},
                    "center_y": {"type": "int", "optional": True},
                    "source_entity_id": {"type": "int", "optional": True},
                    "source_x": {"type": "int", "optional": True},
                    "source_y": {"type": "int", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "radius": {"type": "int"},
                    "material": {"type": "str", "optional": True},
                    "gas_species": {"type": "str", "optional": True},
                    "gas_amount": {"type": "float"},
                    "light_type": {"type": "str", "optional": True},
                    "light_strength": {"type": "float"},
                    "light_spread": {"type": "float"},
                    "force_radius": {"type": "float"},
                    "force_strength": {"type": "float"},
                    "force_lifetime": {"type": "float"},
                    "release_mode": {"type": "str"},
                    "require_empty": {"type": "bool"},
                    "fallback_mode": {"type": "str"},
                    "fallback_radius": {"type": "int"},
                    "potency": {"type": "float"},
                    "stability": {"type": "float"},
                    "label": {"type": "str", "optional": True},
                },
            },
            "observation_target": {
                "fields": observation_target_fields,
                "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                "field_types": {
                    "observer_id": {"type": "int"},
                    "channels": {"type": "str[]"},
                    "center_x": {"type": "int", "optional": True},
                    "center_y": {"type": "int", "optional": True},
                    "width": {"type": "int", "optional": True},
                    "height": {"type": "int", "optional": True},
                    "entity_id": {"type": "int", "optional": True},
                    "pad_cells": {"type": "int"},
                    "label": {"type": "str", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                },
            },
            "readback_request": {
                "fields": readback_request_fields,
                "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                "max_async_window_size": [int(MAX_ASYNC_READBACK_WIDTH), int(MAX_ASYNC_READBACK_HEIGHT)],
                "field_types": {
                    "request_id": {"type": "int", "optional": True},
                    "center_x": {"type": "int", "optional": True},
                    "center_y": {"type": "int", "optional": True},
                    "width": {"type": "int"},
                    "height": {"type": "int"},
                    "channels": {"type": "str[]"},
                    "observer_id": {"type": "int", "optional": True},
                    "label": {"type": "str", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                },
            },
            "material_injector": {
                "fields": material_injector_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "material_alias_fields": ["material"],
                "field_types": {
                    "x": {"type": "int", "optional": True},
                    "y": {"type": "int", "optional": True},
                    "material": {"type": "str"},
                    "radius": {"type": "int"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "temperature_injector": {
                "fields": temperature_injector_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "field_types": {
                    "x": {"type": "int", "optional": True},
                    "y": {"type": "int", "optional": True},
                    "delta": {"type": "float"},
                    "radius": {"type": "int"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "velocity_injector": {
                "fields": velocity_injector_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "carriers": ["cell", "flow", "both"],
                "modes": ["add", "set"],
                "field_types": {
                    "x": {"type": "int", "optional": True},
                    "y": {"type": "int", "optional": True},
                    "velocity": {"type": "float2"},
                    "radius": {"type": "int"},
                    "carrier": {"type": "str"},
                    "mode": {"type": "str"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "gas_injector": {
                "fields": gas_injector_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "field_types": {
                    "x": {"type": "int", "optional": True},
                    "y": {"type": "int", "optional": True},
                    "species": {"type": "str"},
                    "amount": {"type": "float"},
                    "radius": {"type": "int"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "force_injector": {
                "fields": force_injector_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "field_types": {
                    "x": {"type": "float", "optional": True},
                    "y": {"type": "float", "optional": True},
                    "direction": {"type": "float2"},
                    "radius": {"type": "float"},
                    "strength": {"type": "float"},
                    "lifetime": {"type": "float"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "light_injector": {
                "fields": light_injector_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "field_types": {
                    "x": {"type": "int", "optional": True},
                    "y": {"type": "int", "optional": True},
                    "light_type": {"type": "str"},
                    "strength": {"type": "float"},
                    "radius": {"type": "int", "optional": True},
                    "direction": {"type": "float2"},
                    "spread": {"type": "float"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "material": {
                "fields": material_fields,
                "identity_fields": ["material_id", "name"],
                "enums": {
                    "default_phase": [phase.name.lower() for phase in Phase],
                    "collapse_behavior": sorted(COLLAPSE_BEHAVIOR_IDS),
                    "powder_solver_kind": sorted(POWDER_SOLVER_KIND_IDS),
                    "liquid_solver_kind": sorted(LIQUID_SOLVER_KIND_IDS),
                    "falling_island_break_kind": sorted(FALLING_ISLAND_BREAK_KIND_IDS),
                },
            },
            "gas": {
                "fields": gas_fields,
                "identity_fields": ["species_id", "name"],
            },
            "light": {
                "fields": light_fields,
                "identity_fields": ["light_type_id", "name"],
            },
            "optics": {
                "fields": optics_fields,
                "identity_fields": ["material_name", "light_type"],
            },
            "materials_table": {
                "fields": materials_table_fields,
                "record_type": "material",
            },
            "gases_table": {
                "fields": gases_table_fields,
                "record_type": "gas",
            },
            "lights_table": {
                "fields": lights_table_fields,
                "record_type": "light",
            },
            "optics_table": {
                "fields": optics_table_fields,
                "record_type": "optics",
            },
            "reaction_action": {
                "fields": reaction_action_fields,
                "enum_domains": {
                    "reaction_type": [reaction_type.name.lower() for reaction_type in ReactionType],
                    "direction": [direction.value for direction in Direction],
                },
            },
            "pair_reaction_rule": {
                "fields": pair_reaction_rule_fields,
                "phase_domain": [phase.name.lower() for phase in Phase],
            },
            "self_reaction_rule": {
                "fields": self_reaction_rule_fields,
                "phase_domain": [phase.name.lower() for phase in Phase],
            },
            "reactions_table": {
                "fields": reactions_table_fields,
                "action_type": "reaction_action",
                "pair_rule_type": "pair_reaction_rule",
                "self_rule_type": "self_reaction_rule",
                "rule_sets": reaction_table_rule_sets,
            },
            "material_patch": {
                "fields": material_patch_fields,
                "patchable_fields": [field for field in material_fields if field not in {"material_id"}],
            },
            "gas_patch": {
                "fields": gas_patch_fields,
                "patchable_fields": [field for field in gas_fields if field not in {"species_id"}],
            },
            "light_patch": {
                "fields": light_patch_fields,
                "patchable_fields": [field for field in light_fields if field not in {"light_type_id"}],
            },
            "optics_patch": {
                "fields": optics_patch_fields,
                "patchable_fields": ["absorption", "scattering", "refraction"],
            },
            "reaction_action_patch": {
                "fields": reaction_action_patch_fields,
                "patchable_fields": reaction_action_fields,
            },
            "reaction_action_delete_request": {
                "fields": reaction_action_delete_request_fields,
            },
            "reaction_rule_patch": {
                "fields": reaction_rule_patch_fields,
                "rule_sets": reaction_table_rule_sets,
                "pair_rule_type": "pair_reaction_rule",
                "self_rule_type": "self_reaction_rule",
            },
            "reaction_rule_delete_request": {
                "fields": reaction_rule_delete_request_fields,
                "rule_sets": reaction_table_rule_sets,
            },
            "reaction_table_append": {
                "fields": reactions_table_fields,
                "action_type": "reaction_action",
                "pair_rule_type": "pair_reaction_rule",
                "self_rule_type": "self_reaction_rule",
                "rule_sets": reaction_table_rule_sets,
            },
            "reaction_table_replace": {
                "fields": reactions_table_fields,
                "action_type": "reaction_action",
                "pair_rule_type": "pair_reaction_rule",
                "self_rule_type": "self_reaction_rule",
                "rule_sets": reaction_table_rule_sets,
                "replace_semantics": "replace_all",
            },
            "gas_species_runtime": {
                "fields": gas_species_runtime_fields,
            },
            "heat_phase_target": {
                "fields": heat_phase_target_fields,
            },
            "heat_boil_target": {
                "fields": heat_boil_target_fields,
            },
            "heat_condense_target": {
                "fields": heat_condense_target_fields,
            },
            "collapse_component": {
                "fields": collapse_component_fields,
                "bbox_type": "cell_rect",
            },
            "pending_displaced_cell": {
                "fields": pending_displaced_cell_fields,
            },
            "powder_reservation": {
                "fields": powder_reservation_fields,
            },
            "island_reservation": {
                "fields": island_reservation_fields,
            },
            "gas_runtime": {
                "fields": gas_runtime_fields,
                "species_runtime_type": "gas_species_runtime",
            },
            "heat_runtime": {
                "fields": heat_runtime_fields,
                "phase_target_type": "heat_phase_target",
                "boil_target_type": "heat_boil_target",
                "condense_target_type": "heat_condense_target",
            },
            "liquid_runtime": {
                "fields": liquid_runtime_fields,
            },
            "reaction_runtime": {
                "fields": reaction_runtime_fields,
                "stage_names": reaction_table_rule_sets,
            },
            "collapse_runtime": {
                "fields": collapse_runtime_fields,
                "collapsed_component_type": "collapse_component",
            },
            "optics_runtime": {
                "fields": optics_runtime_fields,
                "emitter_type": "emitter",
            },
            "active_runtime": {
                "fields": active_runtime_fields,
                "pending_displaced_cell_type": "pending_displaced_cell",
            },
            "motion_runtime": {
                "fields": motion_runtime_fields,
                "powder_reservation_type": "powder_reservation",
                "island_reservation_type": "island_reservation",
            },
            "control_state": {
                "fields": control_state_fields,
                "field_types": {
                    "paused": {"type": "bool"},
                    "speed": {"type": "float"},
                    "single_step": {"type": "bool"},
                },
            },
            "readback_status": {
                "fields": readback_status_fields,
                "field_types": {
                    "request_id": {"type": "int"},
                    "status": {"type": "str"},
                },
            },
            "readback_cancel_request": {
                "fields": readback_cancel_request_fields,
                "field_types": {
                    "request_id": {"type": "int"},
                },
            },
            "readback_cancel_result": {
                "fields": readback_cancel_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "request_id": {"type": "int"},
                    "status": {"type": "str"},
                },
            },
            "frame_pending_state": {
                "fields": frame_pending_state_fields,
                "field_types": {
                    "pending": {"type": "int"},
                    "submission_ids": {"type": "int[]"},
                },
            },
            "pending_frame_details": {
                "fields": ["pending", "frames"],
                "field_types": {
                    "pending": {"type": "int"},
                    "frames": {"type": "pending_frame_detail[]"},
                },
            },
            "frame_state": {
                "fields": frame_state_fields,
                "field_types": {
                    "pending": {"type": "int"},
                    "pending_submission_ids": {"type": "int[]"},
                    "ready": {"type": "int"},
                    "ready_submission_ids": {"type": "int[]"},
                    "canceled_submission_ids": {"type": "int[]"},
                },
            },
            "frame_submission_status": {
                "fields": frame_submission_status_fields,
                "field_types": {
                    "submission_id": {"type": "int"},
                    "status": {"type": "str"},
                },
            },
            "frame_cancel_request": {
                "fields": frame_cancel_request_fields,
                "field_types": {
                    "submission_id": {"type": "int"},
                },
            },
            "frame_cancel_result": {
                "fields": frame_cancel_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "submission_id": {"type": "int"},
                    "status": {"type": "str"},
                    "pending_frames": {"type": "int"},
                },
            },
            "frame_cancel_all_result": {
                "fields": frame_cancel_all_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "canceled_submission_ids": {"type": "int[]"},
                    "pending_frames": {"type": "int"},
                },
            },
            "queued_mutation_result": {
                "fields": queued_mutation_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                },
            },
            "deferred_readback_request_result": {
                "fields": deferred_readback_request_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                    "request_id": {"type": "int"},
                },
            },
            "deferred_observation_request_result": {
                "fields": deferred_observation_request_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                    "request_id": {"type": "int"},
                },
            },
            "deferred_frame_submit_ack": {
                "fields": deferred_frame_submit_ack_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_frames": {"type": "int"},
                    "submission_id": {"type": "int"},
                },
            },
            "deferred_controller_state_result": {
                "fields": deferred_controller_state_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_frames": {"type": "int"},
                    "submission_id": {"type": "int"},
                },
            },
            "material_fill_request": {
                "fields": material_fill_request_fields,
                "optional_fields": ["immediate", *inline_target_query_optional_fields],
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "material_alias_fields": ["material"],
                "field_types": {
                    "x": {"type": "int", "optional": True},
                    "y": {"type": "int", "optional": True},
                    "width": {"type": "int"},
                    "height": {"type": "int"},
                    "material": {"type": "str"},
                    "immediate": {"type": "bool", "optional": True},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "paging_focus_request": {
                "fields": paging_focus_request_fields,
                "target_query_fields": target_query_fields,
                "supports_inline_target_queries": True,
                "field_types": {
                    "x": {"type": "int"},
                    "y": {"type": "int"},
                    "target_query_id": {"type": "str", "optional": True},
                    "target_dx": {"type": "int"},
                    "target_dy": {"type": "int"},
                    "target_queries": {"type": "target_query[]", "optional": True},
                },
            },
            "paging_focus_result": {
                "fields": paging_focus_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                    "target_center": {"type": "cell_xy", "optional": True},
                },
            },
            "page_store_has_result": {
                "fields": page_store_has_result_fields,
                "field_types": {
                    "stored": {"type": "bool"},
                },
            },
            "page_store_apply_result": {
                "fields": page_store_apply_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "stored": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                },
            },
            "page_store_clear_result": {
                "fields": page_store_clear_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "cleared": {"type": "int"},
                    "stored_stripes": {"type": "int"},
                },
            },
            "page_stripe_apply_result": {
                "fields": page_stripe_apply_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                },
            },
            "speed_request": {
                "fields": speed_request_fields,
                "field_types": {
                    "speed": {"type": "float"},
                },
            },
            "pause_result": {
                "fields": pause_result_fields,
                "field_types": {
                    "paused": {"type": "bool"},
                },
            },
            "resume_result": {
                "fields": resume_result_fields,
                "field_types": {
                    "paused": {"type": "bool"},
                },
            },
            "step_result": {
                "fields": step_result_fields,
                "field_types": {
                    "single_step": {"type": "bool"},
                },
            },
            "speed_result": {
                "fields": speed_result_fields,
                "field_types": {
                    "speed": {"type": "float"},
                },
            },
            "control_reset_result": {
                "fields": control_reset_result_fields,
                "field_types": {
                    "ok": {"type": "bool"},
                    "queued": {"type": "bool"},
                    "pending_commands": {"type": "int"},
                },
            },
            "frame_input": {
                "field_order": frame_input_field_order,
                "fields": {
                    "submission_id": {"type": "int", "optional": True},
                    "focus_center": {"type": "cell_xy", "optional": True},
                    "controller_state": {"type": "json", "optional": True},
                    "controller_state_provided": {"type": "bool", "optional": True},
                    "entities": {"type": "entity_state[]", "optional": True, "replace_semantics": "replace_all"},
                    "entity_placeholders": {
                        "type": "entity_placeholder[]",
                        "optional": True,
                        "replace_semantics": "replace_all",
                    },
                    "force_sources": {
                        "type": "force_source[]",
                        "optional": True,
                        "fields": force_source_fields,
                        "replace_semantics": "replace_all",
                    },
                    "emitters": {
                        "type": "emitter[]",
                        "optional": True,
                        "fields": emitter_fields,
                        "replace_semantics": "replace_all",
                    },
                    "target_queries": {"type": "target_query[]", "optional": True},
                    "change_intents": {"type": "change_intent[]", "optional": True},
                    "carrier_intents": {"type": "carrier_intent[]", "optional": True},
                    "observation_targets": {"type": "observation_target[]", "optional": True},
                    "readback_requests": {"type": "readback_request[]", "optional": True},
                    "commands": {"type": "world_command[]", "optional": True},
                },
                "nested_types": {
                    "entity_state": {
                        "fields": entity_state_fields,
                        "material_alias_fields": ["placeholder_material"],
                        "field_types": {
                            "entity_id": {"type": "int"},
                            "x": {"type": "int"},
                            "y": {"type": "int"},
                            "width": {"type": "int"},
                            "height": {"type": "int"},
                            "velocity_xy": {"type": "float2"},
                            "facing_xy": {"type": "float2", "optional": True},
                            "placeholder_material": {"type": "str"},
                            "tags": {"type": "str[]"},
                            "observe_channels": {"type": "str[]"},
                            "observe_pad_cells": {"type": "int"},
                            "observe_width": {"type": "int", "optional": True},
                            "observe_height": {"type": "int", "optional": True},
                            "observe_label": {"type": "str", "optional": True},
                        },
                    },
                    "entity_placeholder": {
                        "fields": entity_placeholder_fields,
                        "material_alias_fields": ["material"],
                        "field_types": {
                            "entity_id": {"type": "int"},
                            "x": {"type": "int"},
                            "y": {"type": "int"},
                            "width": {"type": "int"},
                            "height": {"type": "int"},
                            "material": {"type": "str"},
                        },
                    },
                    "force_source": {
                        "fields": force_source_fields,
                        "field_types": {
                            "x": {"type": "float"},
                            "y": {"type": "float"},
                            "direction": {"type": "float2"},
                            "radius": {"type": "float"},
                            "strength": {"type": "float"},
                            "lifetime": {"type": "float"},
                        },
                    },
                    "emitter": {
                        "fields": emitter_fields,
                        "field_types": {
                            "x": {"type": "int"},
                            "y": {"type": "int"},
                            "light_type": {"type": "str"},
                            "strength": {"type": "float"},
                            "radius": {"type": "int"},
                            "direction": {"type": "float2"},
                            "spread": {"type": "float"},
                        },
                    },
                    "target_query": {
                        "fields": target_query_fields,
                        "directions": [*CARDINAL_DIRECTION_VECTORS.keys(), "forward", "backward"],
                        "distance_hints": sorted(TARGET_QUERY_DISTANCE_HINT_CELLS),
                        "terrain_anchor_filters": sorted(TERRAIN_ANCHOR_FILTERS),
                        "ignored_anchor_filters": sorted(IGNORED_ANCHOR_FILTERS),
                        "field_types": {
                            "query_id": {"type": "str"},
                            "anchor_filters": {"type": "str[]"},
                            "source_entity_id": {"type": "int", "optional": True},
                            "source_x": {"type": "int", "optional": True},
                            "source_y": {"type": "int", "optional": True},
                            "anchor_entity_id": {"type": "int", "optional": True},
                            "direction": {"type": "str", "optional": True},
                            "distance_cells": {"type": "int"},
                            "distance_meters": {"type": "float", "optional": True},
                            "distance_hint": {"type": "str", "optional": True},
                            "require_empty": {"type": "bool"},
                            "search_radius": {"type": "int"},
                            "label": {"type": "str", "optional": True},
                        },
                    },
                    "change_intent": {
                        "fields": change_intent_fields,
                        "velocity_carriers": ["cell", "flow", "both"],
                        "velocity_modes": ["add", "set"],
                        "fallback_modes": ["nearest_empty", "source"],
                        "material_alias_fields": ["material"],
                        "field_types": {
                            "intent_id": {"type": "str"},
                            "target_query_id": {"type": "str", "optional": True},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                            "radius": {"type": "int"},
                            "material": {"type": "str", "optional": True},
                            "temperature_delta": {"type": "float"},
                            "velocity": {"type": "float2", "optional": True},
                            "velocity_carrier": {"type": "str"},
                            "velocity_mode": {"type": "str"},
                            "require_empty": {"type": "bool"},
                            "fallback_mode": {"type": "str"},
                            "fallback_radius": {"type": "int"},
                            "potency": {"type": "float"},
                            "stability": {"type": "float"},
                            "label": {"type": "str", "optional": True},
                        },
                    },
                    "carrier_intent": {
                        "fields": carrier_intent_fields,
                        "kinds": ["material", "gas", "light", "force"],
                        "release_modes": ["impact", "beam", "projectile"],
                        "fallback_modes": ["nearest_empty", "source"],
                        "material_alias_fields": ["material"],
                        "field_types": {
                            "intent_id": {"type": "str"},
                            "kind": {"type": "str"},
                            "target_query_id": {"type": "str", "optional": True},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "source_entity_id": {"type": "int", "optional": True},
                            "source_x": {"type": "int", "optional": True},
                            "source_y": {"type": "int", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                            "radius": {"type": "int"},
                            "material": {"type": "str", "optional": True},
                            "gas_species": {"type": "str", "optional": True},
                            "gas_amount": {"type": "float"},
                            "light_type": {"type": "str", "optional": True},
                            "light_strength": {"type": "float"},
                            "light_spread": {"type": "float"},
                            "force_radius": {"type": "float"},
                            "force_strength": {"type": "float"},
                            "force_lifetime": {"type": "float"},
                            "release_mode": {"type": "str"},
                            "require_empty": {"type": "bool"},
                            "fallback_mode": {"type": "str"},
                            "fallback_radius": {"type": "int"},
                            "potency": {"type": "float"},
                            "stability": {"type": "float"},
                            "label": {"type": "str", "optional": True},
                        },
                    },
                    "observation_target": {
                        "fields": observation_target_fields,
                        "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                        "field_types": {
                            "observer_id": {"type": "int"},
                            "channels": {"type": "str[]"},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "width": {"type": "int", "optional": True},
                            "height": {"type": "int", "optional": True},
                            "entity_id": {"type": "int", "optional": True},
                            "pad_cells": {"type": "int"},
                            "label": {"type": "str", "optional": True},
                            "target_query_id": {"type": "str", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                        },
                    },
                    "readback_request": {
                        "fields": readback_request_fields,
                        "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                        "max_async_window_size": [int(MAX_ASYNC_READBACK_WIDTH), int(MAX_ASYNC_READBACK_HEIGHT)],
                        "field_types": {
                            "request_id": {"type": "int", "optional": True},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "width": {"type": "int"},
                            "height": {"type": "int"},
                            "channels": {"type": "str[]"},
                            "observer_id": {"type": "int", "optional": True},
                            "label": {"type": "str", "optional": True},
                            "target_query_id": {"type": "str", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                        },
                    },
                    "world_command": {
                        "fields": ["kind", "payload"],
                        "public_kinds": public_world_command_kinds,
                        "target_query_overlay_fields": ["target_query_id", "target_dx", "target_dy"],
                        "material_alias_fields_by_kind": world_command_material_alias_fields_by_kind,
                        "collection_item_types_by_kind": world_command_collection_item_types_by_kind,
                        "field_types": {
                            "kind": {"type": "str"},
                            "payload": {"type": "json"},
                        },
                    },
                },
            },
            "pending_frame_detail": {
                "field_order": pending_frame_detail_field_order,
                "fields": {
                    "submission_id": {"type": "int", "optional": True},
                    "focus_center": {"type": "cell_xy", "optional": True},
                    "controller_state": {"type": "json", "optional": True},
                    "controller_state_provided": {"type": "bool", "optional": True},
                    "entities": {"type": "entity_state[]", "optional": True, "replace_semantics": "replace_all"},
                    "entity_placeholders": {
                        "type": "entity_placeholder[]",
                        "optional": True,
                        "replace_semantics": "replace_all",
                    },
                    "force_sources": {
                        "type": "force_source[]",
                        "optional": True,
                        "fields": force_source_fields,
                        "replace_semantics": "replace_all",
                    },
                    "emitters": {
                        "type": "emitter[]",
                        "optional": True,
                        "fields": emitter_fields,
                        "replace_semantics": "replace_all",
                    },
                    "target_queries": {"type": "target_query[]", "optional": True},
                    "change_intents": {"type": "change_intent[]", "optional": True},
                    "carrier_intents": {"type": "carrier_intent[]", "optional": True},
                    "observation_targets": {"type": "observation_target[]", "optional": True},
                    "readback_requests": {"type": "readback_request[]", "optional": True},
                    "commands": {"type": "world_command[]", "optional": True},
                    "preview": {"type": "frame_preview"},
                },
                "nested_types": {
                    "entity_state": {
                        "fields": entity_state_fields,
                        "material_alias_fields": ["placeholder_material"],
                        "field_types": {
                            "entity_id": {"type": "int"},
                            "x": {"type": "int"},
                            "y": {"type": "int"},
                            "width": {"type": "int"},
                            "height": {"type": "int"},
                            "velocity_xy": {"type": "float2"},
                            "facing_xy": {"type": "float2", "optional": True},
                            "placeholder_material": {"type": "str"},
                            "tags": {"type": "str[]"},
                            "observe_channels": {"type": "str[]"},
                            "observe_pad_cells": {"type": "int"},
                            "observe_width": {"type": "int", "optional": True},
                            "observe_height": {"type": "int", "optional": True},
                            "observe_label": {"type": "str", "optional": True},
                        },
                    },
                    "entity_placeholder": {
                        "fields": entity_placeholder_fields,
                        "material_alias_fields": ["material"],
                        "field_types": {
                            "entity_id": {"type": "int"},
                            "x": {"type": "int"},
                            "y": {"type": "int"},
                            "width": {"type": "int"},
                            "height": {"type": "int"},
                            "material": {"type": "str"},
                        },
                    },
                    "force_source": {
                        "fields": force_source_fields,
                        "field_types": {
                            "x": {"type": "float"},
                            "y": {"type": "float"},
                            "direction": {"type": "float2"},
                            "radius": {"type": "float"},
                            "strength": {"type": "float"},
                            "lifetime": {"type": "float"},
                        },
                    },
                    "emitter": {
                        "fields": emitter_fields,
                        "field_types": {
                            "x": {"type": "int"},
                            "y": {"type": "int"},
                            "light_type": {"type": "str"},
                            "strength": {"type": "float"},
                            "radius": {"type": "int"},
                            "direction": {"type": "float2"},
                            "spread": {"type": "float"},
                        },
                    },
                    "target_query": {
                        "fields": target_query_fields,
                        "directions": [*CARDINAL_DIRECTION_VECTORS.keys(), "forward", "backward"],
                        "distance_hints": sorted(TARGET_QUERY_DISTANCE_HINT_CELLS),
                        "terrain_anchor_filters": sorted(TERRAIN_ANCHOR_FILTERS),
                        "ignored_anchor_filters": sorted(IGNORED_ANCHOR_FILTERS),
                        "field_types": {
                            "query_id": {"type": "str"},
                            "anchor_filters": {"type": "str[]"},
                            "source_entity_id": {"type": "int", "optional": True},
                            "source_x": {"type": "int", "optional": True},
                            "source_y": {"type": "int", "optional": True},
                            "anchor_entity_id": {"type": "int", "optional": True},
                            "direction": {"type": "str", "optional": True},
                            "distance_cells": {"type": "int"},
                            "distance_meters": {"type": "float", "optional": True},
                            "distance_hint": {"type": "str", "optional": True},
                            "require_empty": {"type": "bool"},
                            "search_radius": {"type": "int"},
                            "label": {"type": "str", "optional": True},
                        },
                    },
                    "change_intent": {
                        "fields": change_intent_fields,
                        "velocity_carriers": ["cell", "flow", "both"],
                        "velocity_modes": ["add", "set"],
                        "fallback_modes": ["nearest_empty", "source"],
                        "material_alias_fields": ["material"],
                        "field_types": {
                            "intent_id": {"type": "str"},
                            "target_query_id": {"type": "str", "optional": True},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                            "radius": {"type": "int"},
                            "material": {"type": "str", "optional": True},
                            "temperature_delta": {"type": "float"},
                            "velocity": {"type": "float2", "optional": True},
                            "velocity_carrier": {"type": "str"},
                            "velocity_mode": {"type": "str"},
                            "require_empty": {"type": "bool"},
                            "fallback_mode": {"type": "str"},
                            "fallback_radius": {"type": "int"},
                            "potency": {"type": "float"},
                            "stability": {"type": "float"},
                            "label": {"type": "str", "optional": True},
                        },
                    },
                    "carrier_intent": {
                        "fields": carrier_intent_fields,
                        "kinds": ["material", "gas", "light", "force"],
                        "release_modes": ["impact", "beam", "projectile"],
                        "fallback_modes": ["nearest_empty", "source"],
                        "material_alias_fields": ["material"],
                        "field_types": {
                            "intent_id": {"type": "str"},
                            "kind": {"type": "str"},
                            "target_query_id": {"type": "str", "optional": True},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "source_entity_id": {"type": "int", "optional": True},
                            "source_x": {"type": "int", "optional": True},
                            "source_y": {"type": "int", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                            "radius": {"type": "int"},
                            "material": {"type": "str", "optional": True},
                            "gas_species": {"type": "str", "optional": True},
                            "gas_amount": {"type": "float"},
                            "light_type": {"type": "str", "optional": True},
                            "light_strength": {"type": "float"},
                            "light_spread": {"type": "float"},
                            "force_radius": {"type": "float"},
                            "force_strength": {"type": "float"},
                            "force_lifetime": {"type": "float"},
                            "release_mode": {"type": "str"},
                            "require_empty": {"type": "bool"},
                            "fallback_mode": {"type": "str"},
                            "fallback_radius": {"type": "int"},
                            "potency": {"type": "float"},
                            "stability": {"type": "float"},
                            "label": {"type": "str", "optional": True},
                        },
                    },
                    "observation_target": {
                        "fields": observation_target_fields,
                        "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                        "field_types": {
                            "observer_id": {"type": "int"},
                            "channels": {"type": "str[]"},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "width": {"type": "int", "optional": True},
                            "height": {"type": "int", "optional": True},
                            "entity_id": {"type": "int", "optional": True},
                            "pad_cells": {"type": "int"},
                            "label": {"type": "str", "optional": True},
                            "target_query_id": {"type": "str", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                        },
                    },
                    "readback_request": {
                        "fields": readback_request_fields,
                        "allowed_channels": list(READBACK_ALLOWED_CHANNELS),
                        "max_async_window_size": [int(MAX_ASYNC_READBACK_WIDTH), int(MAX_ASYNC_READBACK_HEIGHT)],
                        "field_types": {
                            "request_id": {"type": "int", "optional": True},
                            "center_x": {"type": "int", "optional": True},
                            "center_y": {"type": "int", "optional": True},
                            "width": {"type": "int"},
                            "height": {"type": "int"},
                            "channels": {"type": "str[]"},
                            "observer_id": {"type": "int", "optional": True},
                            "label": {"type": "str", "optional": True},
                            "target_query_id": {"type": "str", "optional": True},
                            "target_dx": {"type": "int"},
                            "target_dy": {"type": "int"},
                        },
                    },
                    "world_command": {
                        "fields": ["kind", "payload"],
                        "public_kinds": public_world_command_kinds,
                        "target_query_overlay_fields": ["target_query_id", "target_dx", "target_dy"],
                        "material_alias_fields_by_kind": world_command_material_alias_fields_by_kind,
                        "collection_item_types_by_kind": world_command_collection_item_types_by_kind,
                        "field_types": {
                            "kind": {"type": "str"},
                            "payload": {"type": "json"},
                        },
                    },
                },
            },
            "frame_preview": {
                "field_order": frame_preview_field_order,
                "fields": {
                    "controller_state": {"type": "json"},
                    "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                    "resolved_change_intents": {"type": "resolved_change_intent{}", "key": "intent_id"},
                    "resolved_carrier_intents": {"type": "resolved_carrier_intent{}", "key": "intent_id"},
                    "resolved_commands": {"type": "world_command[]"},
                    "observation_requests": {"type": "readback_request[]"},
                    "observation_plans": {"type": "observation_plan[]"},
                    "readback_requests": {"type": "readback_request[]"},
                    "readback_plans": {"type": "readback_plan[]"},
                    "bridge_frame_snapshot": {"type": "bridge_frame_snapshot"},
                    "paging_updates": {"type": "page_stripe_update[]"},
                    "placeholder_count": {"type": "int"},
                },
            },
            "frame_output": {
                "field_order": frame_output_field_order,
                "fields": {
                    "frame_id": {"type": "int"},
                    "submission_id": {"type": "int", "optional": True},
                    "controller_state": {"type": "json"},
                    "consumed_readbacks": {"type": "readback_result[]"},
                    "resolved_targets": {"type": "resolved_target{}", "key": "query_id"},
                    "resolved_change_intents": {"type": "resolved_change_intent{}", "key": "intent_id"},
                    "resolved_carrier_intents": {"type": "resolved_carrier_intent{}", "key": "intent_id"},
                    "observations": {"type": "observation_result{}", "key": "observer_id"},
                    "entity_feedback": {"type": "entity_feedback{}", "key": "entity_id"},
                    "paging_updates": {"type": "page_stripe_update[]"},
                    "observation_plans": {"type": "observation_plan[]"},
                    "readback_plans": {"type": "readback_plan[]"},
                    "bridge_upload_snapshot": {"type": "bridge_upload_snapshot"},
                    "bridge_frame_snapshot": {"type": "bridge_frame_snapshot"},
                    "queued_observations": {"type": "int"},
                    "queued_readbacks": {"type": "int"},
                    "queued_commands": {"type": "int"},
                    "placeholder_count": {"type": "int"},
                },
            },
            "frame_output_poll": {
                "fields": ["ready", "status", "output"],
                "field_types": {
                    "ready": {"type": "bool"},
                    "status": {"type": "str"},
                    "output": {"type": "frame_output", "optional": True},
                },
            },
            "frame_output_ready": {
                "fields": ["ready", "outputs"],
                "field_types": {
                    "ready": {"type": "int"},
                    "outputs": {"type": "frame_output[]"},
                },
            },
            "frame_output_poll_all": {
                "fields": ["outputs"],
                "field_types": {
                    "outputs": {"type": "frame_output[]"},
                },
            },
            "frame_cycle_result": {
                "fields": ["ok", "applied", "queued", "pending_frames", "submission_id", "preview", "result"],
                "preview_type": "frame_preview",
                "result_type": "frame_output",
                "result_optional_when_unapplied": True,
                "result_optional_when_deferred": True,
                "submission_id_optional_when_unapplied": True,
                "queueing": "deferred_frame_when_applied",
            },
            "frame_submit_result": {
                "fields": ["ok", "queued", "pending_frames", "submission_id", "preview"],
                "preview_type": "frame_preview",
                "submission_id_optional": False,
                "queueing": "deferred_frame",
            },
            "page_stripe_update": {
                "fields": [
                    "axis",
                    "world_start",
                    "world_end",
                    "buffer_start",
                    "buffer_end",
                    "kind",
                    "cross_world_start",
                    "cross_world_end",
                ],
                "axes": ["x", "y"],
                "kinds": ["save", "load"],
                "field_types": {
                    "axis": {"type": "str"},
                    "world_start": {"type": "int"},
                    "world_end": {"type": "int"},
                    "buffer_start": {"type": "int"},
                    "buffer_end": {"type": "int"},
                    "kind": {"type": "str"},
                    "cross_world_start": {"type": "int"},
                    "cross_world_end": {"type": "int"},
                },
            },
            "paging_state": {
                "fields": paging_state_fields,
                "origin_type": "cell_xy",
                "buffer_origin_type": "cell_xy",
                "active_bounds_type": "cell_rect",
                "buffer_size_type": "cell_wh",
                "active_size_type": "cell_wh",
            },
            "page_stripe_payload": {
                "fields": ["meta", "cell", "runtime", "gas"],
            },
            "page_stripe_apply": {
                "fields": ["update", "payload", "immediate"],
                "update_type": "page_stripe_update",
                "payload_type": "page_stripe_payload",
                "immediate_optional": True,
                "field_types": {
                    "update": {"type": "page_stripe_update"},
                    "payload": {"type": "page_stripe_payload"},
                    "immediate": {"type": "bool", "optional": True},
                },
            },
            "page_store_state": {
                "fields": ["stored_stripes", "key_listing_supported", "stripe_keys"],
                "stripe_key_type": "page_store_key",
                "field_types": {
                    "stored_stripes": {"type": "int"},
                    "key_listing_supported": {"type": "bool"},
                    "stripe_keys": {"type": "page_store_key[]"},
                },
            },
            "page_store_key": {
                "fields": ["axis", "world_start", "world_end", "cross_world_start", "cross_world_end"],
                "axes": ["x", "y"],
                "field_types": {
                    "axis": {"type": "str"},
                    "world_start": {"type": "int"},
                    "world_end": {"type": "int"},
                    "cross_world_start": {"type": "int"},
                    "cross_world_end": {"type": "int"},
                },
            },
            "page_store_entry": {
                "fields": ["key", "payload"],
                "key_type": "page_store_key",
                "payload_type": "page_stripe_payload",
                "field_types": {
                    "key": {"type": "page_store_key"},
                    "payload": {"type": "page_stripe_payload"},
                },
            },
            "page_store_export": {
                "fields": ["stored_stripes", "key_listing_supported", "entries"],
                "entry_type": "page_store_entry",
                "field_types": {
                    "stored_stripes": {"type": "int"},
                    "key_listing_supported": {"type": "bool"},
                    "entries": {"type": "page_store_entry[]"},
                },
            },
            "page_store_import": {
                "fields": ["entries", "clear"],
                "entry_type": "page_store_entry",
                "clear_optional": True,
                "field_types": {
                    "entries": {"type": "page_store_entry[]"},
                    "clear": {"type": "bool", "optional": True},
                },
            },
            "page_store_capture_result": {
                "fields": ["ok", "stored_stripes", "payload"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "stored_stripes": {"type": "int"},
                    "payload": {"type": "page_stripe_payload"},
                },
            },
            "page_store_load_result": {
                "fields": ["ok", "stored", "payload"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "stored": {"type": "bool"},
                    "payload": {"type": "page_stripe_payload", "optional": True},
                },
            },
            "page_store_save_result": {
                "fields": ["ok", "stored_stripes"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "stored_stripes": {"type": "int"},
                },
            },
            "page_stripe_capture_result": {
                "fields": ["ok", "payload"],
                "field_types": {
                    "ok": {"type": "bool"},
                    "payload": {"type": "page_stripe_payload"},
                },
            },
            "page_store_import_result": {
                "fields": ["cleared", "imported", "stored_stripes"],
                "field_types": {
                    "cleared": {"type": "int"},
                    "imported": {"type": "int"},
                    "stored_stripes": {"type": "int"},
                },
            },
            "pending_commands": {
                "fields": pending_commands_fields,
                "field_types": {
                    "pending": {"type": "int"},
                    "commands": {"type": "world_command[]"},
                },
            },
            "tables": {
                "materials": {
                    "record_fields": material_fields,
                    "identity_fields": ["material_id", "name"],
                    "update_semantics": "merge_by_material_id",
                    "patch_identity_field": "name",
                    "patchable_fields": [field for field in material_fields if field not in {"material_id"}],
                    "material_alias_fields": [
                        "name",
                        "collapse_generation",
                        "powder_generation",
                        "melt_to_material",
                        "freeze_to_material",
                    ],
                    "enums": {
                        "default_phase": [phase.name.lower() for phase in Phase],
                        "collapse_behavior": sorted(COLLAPSE_BEHAVIOR_IDS),
                        "powder_solver_kind": sorted(POWDER_SOLVER_KIND_IDS),
                        "liquid_solver_kind": sorted(LIQUID_SOLVER_KIND_IDS),
                        "falling_island_break_kind": sorted(FALLING_ISLAND_BREAK_KIND_IDS),
                    },
                },
                "gases": {
                    "record_fields": gas_fields,
                    "identity_fields": ["species_id", "name"],
                    "update_semantics": "merge_by_species_id",
                    "patch_identity_field": "name",
                    "patchable_fields": [field for field in gas_fields if field not in {"species_id"}],
                    "material_alias_fields": ["condense_to_material"],
                },
                "lights": {
                    "record_fields": light_fields,
                    "identity_fields": ["light_type_id", "name"],
                    "update_semantics": "merge_by_light_type_id",
                    "patch_identity_field": "name",
                    "patchable_fields": [field for field in light_fields if field not in {"light_type_id"}],
                },
                "optics": {
                    "record_fields": optics_fields,
                    "identity_fields": ["material_name", "light_type"],
                    "update_semantics": "merge_by_material_name_and_light_type",
                    "patch_identity_fields": ["material_name", "light_type"],
                    "patchable_fields": ["absorption", "scattering", "refraction"],
                    "material_alias_fields": ["material_name"],
                },
                "reactions": {
                    "action_fields": reaction_action_fields,
                    "action_patch_identity_field": "index",
                    "action_patchable_fields": reaction_action_fields,
                    "action_material_alias_fields": ["target_material", "emit_material"],
                    "append_rule_sets": [
                        "material_material",
                        "material_gas",
                        "material_light",
                        "gas_gas",
                        "gas_light",
                        "self_rules",
                    ],
                    "replace_semantics": "replace_all",
                    "rule_fields": {
                        "pair_rule": pair_reaction_rule_fields,
                        "self_rule": self_reaction_rule_fields,
                    },
                    "rule_material_alias_fields": {
                        "pair_rule": ["lhs_material", "rhs_material"],
                        "self_rule": ["material"],
                    },
                    "enum_domains": {
                        "reaction_type": [reaction_type.name.lower() for reaction_type in ReactionType],
                        "direction": [direction.value for direction in Direction],
                        "phase": [phase.name.lower() for phase in Phase],
                    },
                },
            },
            "materials": [
                {
                    "material_id": int(item["material_id"]),
                    "name": str(item["name"]),
                    "display_name": str(item["display_name"]),
                    "default_phase": int(self._coerce_enum(Phase, item["default_phase"])),
                    "render_group": str(item["render_group"]),
                    "tags": list(item.get("tags", ())),
                }
                for item in material_payload
            ],
            "gases": [
                {
                    "species_id": int(item["species_id"]),
                    "name": str(item["name"]),
                    "display_name": str(item["display_name"]),
                }
                for item in gas_payload
            ],
            "lights": [
                {
                    "light_type_id": int(item["light_type_id"]),
                    "name": str(item["name"]),
                    "display_name": str(item["display_name"]),
                    "default_range": int(item["default_range"]),
                    "dose_channel_id": int(item["dose_channel_id"]),
                    "visual_channel": int(item["visual_channel"]),
                }
                for item in light_payload
            ],
        }

    def serialize_material_table(self) -> list[dict[str, Any]]:
        payload = self._shadow_material_payload()
        return self._normalize_json_payload_value(payload)

    def _serialize_bridge_ndarray(self, name: str, array: np.ndarray) -> dict[str, Any]:
        if array.dtype.names is not None:
            rows = [
                {
                    str(field_name): self._normalize_json_payload_value(row[field_name])
                    for field_name in array.dtype.names
                }
                for row in array
            ]
            return {
                "name": str(name),
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "structured": True,
                "field_names": [str(field_name) for field_name in array.dtype.names],
                "row_count": int(array.shape[0]) if array.ndim > 0 else 0,
                "rows": rows,
            }

        utf8: str | None = None
        if array.dtype == np.uint8 and array.ndim == 1:
            try:
                utf8 = bytes(np.ascontiguousarray(array).tolist()).decode("utf-8")
            except UnicodeDecodeError:
                utf8 = None
        return {
            "name": str(name),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "structured": False,
            "field_names": [],
            "row_count": int(array.shape[0]) if array.ndim > 0 else 0,
            "values": self._normalize_json_payload_value(array),
            "utf8": utf8,
        }

    @staticmethod
    def _bridge_row_count(array: np.ndarray) -> int:
        return int(array.shape[0]) if array.ndim > 0 else 0

    @classmethod
    def _normalize_bridge_slice_bounds(cls, array: np.ndarray, *, offset: int = 0, limit: int | None = None) -> tuple[int, int]:
        row_count = cls._bridge_row_count(array)
        start = min(max(0, int(offset)), row_count)
        if limit is None:
            end = row_count
        else:
            end = min(row_count, start + max(0, int(limit)))
        return start, end

    @staticmethod
    def _normalize_bridge_window_bounds(
        array: np.ndarray,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> tuple[int, int, int, int]:
        if array.ndim < 2:
            raise ValueError("bridge shadow buffer window requires at least 2 dimensions")
        width = int(array.shape[-1])
        height = int(array.shape[-2])
        x0 = min(max(0, int(x)), width)
        y0 = min(max(0, int(y)), height)
        if w is None:
            x1 = width
        else:
            x1 = min(width, x0 + max(0, int(w)))
        if h is None:
            y1 = height
        else:
            y1 = min(height, y0 + max(0, int(h)))
        return x0, y0, x1, y1

    def _clamped_gas_window(self, gas_x: int, gas_y: int, width: int, height: int) -> tuple[int, int, int, int]:
        min_gas_x = int(self.paging.origin_x) // int(self.gas_cell_size)
        min_gas_y = int(self.paging.origin_y) // int(self.gas_cell_size)
        max_gas_x = min_gas_x + int(self.gas_width)
        max_gas_y = min_gas_y + int(self.gas_height)
        clamped_gas_x = min_gas_x if self.gas_width <= 0 else max(min_gas_x, min(max_gas_x - 1, int(gas_x)))
        clamped_gas_y = min_gas_y if self.gas_height <= 0 else max(min_gas_y, min(max_gas_y - 1, int(gas_y)))
        span_x = max(0, int(width))
        span_y = max(0, int(height))
        return (
            int(clamped_gas_x),
            int(clamped_gas_y),
            int(min(max_gas_x, clamped_gas_x + span_x)),
            int(min(max_gas_y, clamped_gas_y + span_y)),
        )

    def _bridge_shadow_buffer_coord_space(self, array: np.ndarray) -> str | None:
        if array.ndim < 2:
            return None
        trailing_shape = tuple(int(value) for value in array.shape[-2:])
        if trailing_shape == (int(self.height), int(self.width)):
            return "world"
        if trailing_shape == (int(self.gas_height), int(self.gas_width)):
            return "gas"
        return None

    def _serialize_bridge_ndarray_slice(
        self,
        name: str,
        array: np.ndarray,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        start, end = self._normalize_bridge_slice_bounds(array, offset=offset, limit=limit)
        sliced = array[start:end]
        payload = {
            "name": str(name),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "structured": array.dtype.names is not None,
            "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
            "row_count": self._bridge_row_count(array),
            "offset": int(start),
            "limit": int(end - start if limit is None else max(0, int(limit))),
            "returned_count": int(end - start),
            "slice_shape": [int(value) for value in sliced.shape],
        }
        if array.dtype.names is not None:
            payload["rows"] = [
                {
                    str(field_name): self._normalize_json_payload_value(row[field_name])
                    for field_name in array.dtype.names
                }
                for row in sliced
            ]
            payload["values"] = []
            payload["utf8"] = None
            return payload

        utf8: str | None = None
        if sliced.dtype == np.uint8 and sliced.ndim == 1:
            try:
                utf8 = bytes(np.ascontiguousarray(sliced).tolist()).decode("utf-8")
            except UnicodeDecodeError:
                utf8 = None
        payload["rows"] = []
        payload["values"] = self._normalize_json_payload_value(sliced)
        payload["utf8"] = utf8
        return payload

    def _serialize_bridge_ndarray_window(
        self,
        name: str,
        array: np.ndarray,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        x0, y0, x1, y1 = self._normalize_bridge_window_bounds(array, x=x, y=y, w=w, h=h)
        selection = (slice(None),) * max(0, array.ndim - 2) + (slice(y0, y1), slice(x0, x1))
        window = array[selection]
        payload = {
            "name": str(name),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "structured": array.dtype.names is not None,
            "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
            "row_count": self._bridge_row_count(array),
            "window_origin": [int(x0), int(y0)],
            "requested_size": [max(0, 0 if w is None else int(w)), max(0, 0 if h is None else int(h))],
            "window_size": [int(x1 - x0), int(y1 - y0)],
            "window_axes": [int(array.ndim - 2), int(array.ndim - 1)],
            "returned_shape": [int(value) for value in window.shape],
        }
        if array.dtype.names is not None:
            payload["rows"] = self._normalize_json_payload_value(window)
            payload["values"] = []
            payload["utf8"] = None
            return payload
        payload["rows"] = []
        payload["values"] = self._normalize_json_payload_value(window)
        payload["utf8"] = None
        return payload

    def _serialize_bridge_spatial_window_payload(
        self,
        name: str,
        array: np.ndarray,
        *,
        coord_space: str,
        window_origin: tuple[int, int],
        requested_size: tuple[int, int],
        window_size: tuple[int, int],
        window: np.ndarray,
    ) -> dict[str, Any]:
        payload = {
            "name": str(name),
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "structured": array.dtype.names is not None,
            "field_names": [] if array.dtype.names is None else [str(field_name) for field_name in array.dtype.names],
            "row_count": self._bridge_row_count(array),
            "coord_space": str(coord_space),
            "window_origin": [int(window_origin[0]), int(window_origin[1])],
            "requested_size": [int(requested_size[0]), int(requested_size[1])],
            "window_size": [int(window_size[0]), int(window_size[1])],
            "window_axes": [int(array.ndim - 2), int(array.ndim - 1)],
            "returned_shape": [int(value) for value in window.shape],
        }
        if array.dtype.names is not None:
            payload["rows"] = self._normalize_json_payload_value(window)
            payload["values"] = []
            payload["utf8"] = None
            return payload
        payload["rows"] = []
        payload["values"] = self._normalize_json_payload_value(window)
        payload["utf8"] = None
        return payload

    def serialize_bridge_typed_table(self, name: str) -> dict[str, Any]:
        table = self.bridge.shadow_typed_tables.get(str(name))
        if table is None:
            raise KeyError(name)
        if table.dtype.names is None:
            raise ValueError(f"bridge typed table '{name}' is not a structured array")
        return self._serialize_bridge_ndarray(str(name), table)

    def serialize_bridge_typed_table_slice(self, name: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        table = self.bridge.shadow_typed_tables.get(str(name))
        if table is None:
            raise KeyError(name)
        if table.dtype.names is None:
            raise ValueError(f"bridge typed table '{name}' is not a structured array")
        payload = self._serialize_bridge_ndarray_slice(str(name), table, offset=offset, limit=limit)
        payload.pop("values", None)
        payload.pop("utf8", None)
        return payload

    def serialize_bridge_shadow_buffer(self, name: str) -> dict[str, Any]:
        buffer = self.bridge.shadow_buffers.get(str(name))
        if buffer is None:
            raise KeyError(name)
        if not isinstance(buffer, np.ndarray):
            raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
        payload = self._serialize_bridge_ndarray(str(name), buffer)
        if payload.get("structured"):
            payload.setdefault("values", [])
            payload.setdefault("utf8", None)
        else:
            payload.setdefault("rows", [])
        return payload

    def serialize_bridge_shadow_buffer_slice(self, name: str, *, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        buffer = self.bridge.shadow_buffers.get(str(name))
        if buffer is None:
            raise KeyError(name)
        if not isinstance(buffer, np.ndarray):
            raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
        return self._serialize_bridge_ndarray_slice(str(name), buffer, offset=offset, limit=limit)

    def serialize_bridge_shadow_buffer_window(
        self,
        name: str,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        buffer = self.bridge.shadow_buffers.get(str(name))
        if buffer is None:
            raise KeyError(name)
        if not isinstance(buffer, np.ndarray):
            raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
        return self._serialize_bridge_ndarray_window(str(name), buffer, x=x, y=y, w=w, h=h)

    def serialize_bridge_shadow_buffer_world_window(
        self,
        name: str,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        buffer = self.bridge.shadow_buffers.get(str(name))
        if buffer is None:
            raise KeyError(name)
        if not isinstance(buffer, np.ndarray):
            raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
        coord_space = self._bridge_shadow_buffer_coord_space(buffer)
        if coord_space != "world":
            raise ValueError(f"bridge shadow buffer '{name}' does not use world grid coordinates")
        world_x0, world_y0, world_x1, world_y1 = self._clamped_world_window(int(x), int(y), self.width if w is None else int(w), self.height if h is None else int(h))
        window = self._extract_world_window(
            buffer,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=buffer.ndim - 1,
            y_axis=buffer.ndim - 2,
        )
        return self._serialize_bridge_spatial_window_payload(
            str(name),
            buffer,
            coord_space="world",
            window_origin=(world_x0, world_y0),
            requested_size=(max(0, self.width if w is None else int(w)), max(0, self.height if h is None else int(h))),
            window_size=(world_x1 - world_x0, world_y1 - world_y0),
            window=window,
        )

    def serialize_bridge_shadow_buffer_gas_window(
        self,
        name: str,
        *,
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
    ) -> dict[str, Any]:
        buffer = self.bridge.shadow_buffers.get(str(name))
        if buffer is None:
            raise KeyError(name)
        if not isinstance(buffer, np.ndarray):
            raise TypeError(f"bridge shadow buffer '{name}' is not an ndarray")
        coord_space = self._bridge_shadow_buffer_coord_space(buffer)
        if coord_space != "gas":
            raise ValueError(f"bridge shadow buffer '{name}' does not use gas grid coordinates")
        gas_x0, gas_y0, gas_x1, gas_y1 = self._clamped_gas_window(int(x), int(y), self.gas_width if w is None else int(w), self.gas_height if h is None else int(h))
        window = self._extract_world_window(
            buffer,
            gas_x0,
            gas_y0,
            gas_x1,
            gas_y1,
            x_axis=buffer.ndim - 1,
            y_axis=buffer.ndim - 2,
            gas_grid=True,
        )
        return self._serialize_bridge_spatial_window_payload(
            str(name),
            buffer,
            coord_space="gas",
            window_origin=(gas_x0, gas_y0),
            requested_size=(max(0, self.gas_width if w is None else int(w)), max(0, self.gas_height if h is None else int(h))),
            window_size=(gas_x1 - gas_x0, gas_y1 - gas_y0),
            window=window,
        )

    @staticmethod
    def _decode_bridge_uploaded_command(meta_record: np.ndarray, payload_bytes: np.ndarray) -> dict[str, Any]:
        start = int(meta_record["payload_offset"])
        end = start + int(meta_record["payload_length"])
        return json.loads(payload_bytes[start:end].tobytes().decode("utf-8"))

    @staticmethod
    def _decode_bridge_uploaded_label(meta_record: np.ndarray, label_bytes: np.ndarray) -> str:
        start = int(meta_record["label_offset"])
        end = start + int(meta_record["label_length"])
        return label_bytes[start:end].tobytes().decode("utf-8")

    @staticmethod
    def _decode_bridge_uploaded_page_stripe_section(section_record: np.ndarray, payload_bytes: np.ndarray) -> np.ndarray:
        dtype_map = {
            1: np.uint8,
            2: np.int32,
            3: np.uint32,
            4: np.float32,
        }
        dtype = dtype_map[int(section_record["dtype_code"])]
        ndim = int(section_record["ndim"])
        shape = tuple(int(section_record[f"dim{axis}"]) for axis in range(ndim))
        start = int(section_record["byte_offset"])
        end = start + int(section_record["byte_length"])
        return np.frombuffer(payload_bytes[start:end].tobytes(), dtype=dtype).reshape(shape)

    @staticmethod
    def _set_nested_payload_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        cursor = payload
        for key in path[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[path[-1]] = value

    def serialize_bridge_upload_snapshot(self) -> dict[str, Any]:
        commands_meta = self.bridge.shadow_buffers.get("world_command")
        commands_payload = self.bridge.shadow_buffers.get("world_command_payload")
        readback_meta = self.bridge.shadow_buffers.get("readback_request")
        readback_labels = self.bridge.shadow_buffers.get("readback_request_label")
        stripe_meta = self.bridge.shadow_buffers.get("page_stripe_meta")
        stripe_sections = self.bridge.shadow_buffers.get("page_stripe_section")
        stripe_payload = self.bridge.shadow_buffers.get("page_stripe_payload")
        frame_meta = self.bridge.shadow_buffers.get("frame_meta")

        world_commands = []
        if isinstance(commands_meta, np.ndarray) and isinstance(commands_payload, np.ndarray):
            world_commands = [
                self._normalize_json_payload_value(self._decode_bridge_uploaded_command(record, commands_payload))
                for record in commands_meta
            ]

        readback_requests = []
        if isinstance(readback_meta, np.ndarray) and isinstance(readback_labels, np.ndarray):
            for record in readback_meta:
                channels = [
                    channel
                    for channel, bit in READBACK_CHANNEL_BITS.items()
                    if int(record["channels_mask"]) & int(bit)
                ]
                readback_requests.append(
                    {
                        "request_id": int(record["request_id"]),
                        "center_x": int(record["center_x"]),
                        "center_y": int(record["center_y"]),
                        "width": int(record["width"]),
                        "height": int(record["height"]),
                        "channels_mask": int(record["channels_mask"]),
                        "channels": channels,
                        "observer_id": int(record["observer_id"]),
                        "label": self._decode_bridge_uploaded_label(record, readback_labels),
                    }
                )

        axis_names = {value: key for key, value in PAGE_STRIPE_AXIS_IDS.items()}
        kind_names = {value: key for key, value in PAGE_STRIPE_KIND_IDS.items()}
        field_paths = {field_id: path for field_id, path in PAGE_STRIPE_FIELD_PATHS}
        page_stripes = []
        if (
            isinstance(stripe_meta, np.ndarray)
            and isinstance(stripe_sections, np.ndarray)
            and isinstance(stripe_payload, np.ndarray)
        ):
            for stripe_index, record in enumerate(stripe_meta):
                payload: dict[str, Any] = {}
                start = int(record["section_offset"])
                end = start + int(record["section_count"])
                for section in stripe_sections[start:end]:
                    path = field_paths.get(int(section["field_id"]))
                    if path is None:
                        continue
                    self._set_nested_payload_value(
                        payload,
                        path,
                        self._decode_bridge_uploaded_page_stripe_section(section, stripe_payload),
                    )
                serialized_payload = None
                if payload:
                    serialized_payload = self.serialize_page_stripe_payload(payload)
                page_stripes.append(
                    {
                        "stripe_index": int(stripe_index),
                        "axis": axis_names.get(int(record["axis_id"]), "unknown"),
                        "kind": kind_names.get(int(record["kind_id"]), "unknown"),
                        "world_start": int(record["world_start"]),
                        "world_end": int(record["world_end"]),
                        "buffer_start": int(record["buffer_start"]),
                        "buffer_end": int(record["buffer_end"]),
                        "section_count": int(record["section_count"]),
                        "payload": serialized_payload,
                    }
                )

        frame_meta_rows: list[dict[str, Any]] = []
        if isinstance(frame_meta, np.ndarray) and frame_meta.dtype.names is not None:
            frame_meta_rows = [
                {
                    str(field_name): self._normalize_json_payload_value(row[field_name])
                    for field_name in frame_meta.dtype.names
                }
                for row in frame_meta
            ]

        return {
            "frame_meta": frame_meta_rows,
            "world_commands": world_commands,
            "readback_requests": readback_requests,
            "page_stripes": page_stripes,
        }

    def serialize_bridge_frame_snapshot(self) -> dict[str, Any]:
        snapshot_prepared = bool(
            self._bridge_inputs_prepared
            or self.bridge_frame_commands
            or self.bridge_frame_readback_requests
            or self.bridge_frame_placeholders
            or self.bridge_frame_placeholder_dirty_rects
            or self.bridge_frame_paging_updates
            or self.bridge_frame_page_stripes
        )
        placeholder_dirty_rects = []
        for x0, y0, x1, y1 in self.bridge_frame_placeholder_dirty_rects:
            world_rect = self._buffer_bbox_to_world_bbox((int(x0), int(y0), int(x1), int(y1)))
            placeholder_dirty_rects.append(
                {
                    "buffer_rect": [int(x0), int(y0), int(x1), int(y1)],
                    "world_rect": [int(world_rect[0]), int(world_rect[1]), int(world_rect[2]), int(world_rect[3])],
                }
            )

        page_stripes = [
            {
                "update": self.serialize_page_stripe_update(update),
                "payload": self.serialize_page_stripe_payload(payload),
            }
            for update, payload in self.bridge_frame_page_stripes
        ]

        return {
            "prepared": snapshot_prepared,
            "commands": [self.serialize_world_command(command) for command in self.bridge_frame_commands],
            "command_stages": self._serialize_bridge_index_stages(
                self.bridge_frame_commands,
                stage="staged",
            ),
            "readback_requests": [
                self.serialize_readback_request(request) for request in self.bridge_frame_readback_requests
            ],
            "readback_request_stages": self._serialize_bridge_readback_request_stages(
                self.bridge_frame_readback_requests,
                stage="staged",
            ),
            "placeholders": [
                self.serialize_entity_placeholder_input(placeholder) for placeholder in self.bridge_frame_placeholders
            ],
            "placeholder_stages": self._serialize_bridge_index_stages(
                self.bridge_frame_placeholders,
                stage="staged",
            ),
            "placeholder_dirty_rects": placeholder_dirty_rects,
            "paging_updates": [
                self.serialize_page_stripe_update(update) for update in self.bridge_frame_paging_updates
            ],
            "paging_update_stages": self._serialize_bridge_index_stages(
                self.bridge_frame_paging_updates,
                stage="staged",
            ),
            "page_stripes": page_stripes,
            "page_stripe_stages": self._serialize_bridge_index_stages(
                self.bridge_frame_page_stripes,
                stage="staged",
            ),
        }

    def _serialize_force_source_record(self, force_source: ForceSource) -> dict[str, Any]:
        world_x, world_y = self._force_source_world_position(force_source)
        return {
            "x": float(world_x),
            "y": float(world_y),
            "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
            "radius": float(force_source.radius),
            "strength": float(force_source.strength),
            "lifetime": float(force_source.lifetime),
        }

    def serialize_force_sources(self) -> list[dict[str, Any]]:
        return [self._serialize_force_source_record(force_source) for force_source in self.force_sources]

    def _serialize_emitter_record(self, emitter: dict[str, object]) -> dict[str, object]:
        if "world_origin" in emitter:
            world_x, world_y = (int(emitter["world_origin"][0]), int(emitter["world_origin"][1]))
        else:
            world_x, world_y = self._buffer_to_world_position((int(emitter["origin"][0]), int(emitter["origin"][1])))
        return {
            "x": int(world_x),
            "y": int(world_y),
            "light_type": str(emitter["light_type"]),
            "direction": [float(emitter["direction"][0]), float(emitter["direction"][1])],
            "spread": float(emitter["spread"]),
            "strength": float(emitter["strength"]),
            "radius": int(emitter["range_cells"]),
        }

    def serialize_emitters(self) -> dict[str, list[dict[str, object]]]:
        return {
            "persistent_emitters": [self._serialize_emitter_record(emitter) for emitter in self.persistent_emitters],
            "queued_emitters": [self._serialize_emitter_record(emitter) for emitter in self.emitters],
        }

    def serialize_gas_species_table(self) -> list[dict[str, Any]]:
        payload = self._shadow_gas_species_payload()
        return self._normalize_json_payload_value(payload)

    def serialize_light_type_table(self) -> list[dict[str, Any]]:
        payload = self._shadow_light_type_payload()
        return self._normalize_json_payload_value(payload)

    def serialize_material_optics_table(self) -> list[dict[str, Any]]:
        payload = self._stable_shadow_payload("optics", self._material_optics_table_snapshot_payload)
        return self._normalize_json_payload_value(payload)

    def serialize_reaction_table(self) -> dict[str, object]:
        payload = self._shadow_reaction_payload()
        return self._normalize_json_payload_value(payload)

    def serialize_optics(
        self,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
        *,
        light_type: str | None = None,
    ) -> dict[str, Any]:
        resolved_width = self.width if width is None else max(0, int(width))
        resolved_height = self.height if height is None else max(0, int(height))
        world_x0, world_y0, world_x1, world_y1 = self._clamped_world_window(
            int(x),
            int(y),
            resolved_width,
            resolved_height,
        )
        gas_world_x0, gas_world_y0, gas_world_x1, gas_world_y1 = self._world_gas_window_for_cell_world_rect(
            world_x0,
            world_y0,
            world_x1,
            world_y1,
        )
        light_entries: list[tuple[str, int]] = []
        if light_type is None:
            light_entries = [
                (shadow_name, dose_channel)
                for light_id in range(len(self.light_name_by_id))
                for shadow_name in [self._shadow_light_name(light_id)]
                for dose_channel in [self._shadow_light_dose_channel(light_id)]
                if shadow_name
                and dose_channel is not None
                and 0 <= int(dose_channel) < self.cell_optical_dose.shape[0]
                and 0 <= int(dose_channel) < self.gas_optical_dose.shape[0]
            ]
        else:
            light_id = self._resolve_sanctioned_light_id(light_type)
            if light_id < 0:
                raise KeyError(light_type)
            dose_channel = self._shadow_light_dose_channel(light_id)
            if dose_channel is None:
                raise KeyError(light_type)
            if not (0 <= dose_channel < self.cell_optical_dose.shape[0] and 0 <= dose_channel < self.gas_optical_dose.shape[0]):
                raise KeyError(light_type)
            shadow_light_name = self._shadow_light_name(light_id)
            if shadow_light_name is None:
                raise KeyError(light_type)
            light_entries = [(shadow_light_name, dose_channel)]
        return {
            "origin": [world_x0, world_y0],
            "size": [world_x1 - world_x0, world_y1 - world_y0],
            "gas_origin": [gas_world_x0, gas_world_y0],
            "gas_size": [gas_world_x1 - gas_world_x0, gas_world_y1 - gas_world_y0],
            "visible_illumination": self._extract_world_window(
                self.visible_illumination,
                world_x0,
                world_y0,
                world_x1,
                world_y1,
                x_axis=1,
                y_axis=0,
            )
            .round(4)
            .tolist(),
            "cell_dose": {
                light_name: self._extract_world_window(
                    self.cell_optical_dose[dose_channel],
                    world_x0,
                    world_y0,
                    world_x1,
                    world_y1,
                    x_axis=1,
                    y_axis=0,
                )
                .round(4)
                .tolist()
                for light_name, dose_channel in light_entries
            },
            "gas_dose": {
                light_name: self._extract_world_window(
                    self.gas_optical_dose[dose_channel],
                    gas_world_x0,
                    gas_world_y0,
                    gas_world_x1,
                    gas_world_y1,
                    x_axis=1,
                    y_axis=0,
                    gas_grid=True,
                )
                .round(4)
                .tolist()
                for light_name, dose_channel in light_entries
            },
        }

    def serialize_readback_request(self, request: ReadbackRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "center_x": int(request.center_x) if request.center_x is not None else None,
            "center_y": int(request.center_y) if request.center_y is not None else None,
            "width": int(request.width),
            "height": int(request.height),
            "channels": list(request.channels),
            "observer_id": request.observer_id,
            "label": request.label,
            "target_query_id": request.target_query_id,
            "target_dx": int(request.target_dx),
            "target_dy": int(request.target_dy),
        }

    def _infer_readback_payload_coord_space(
        self,
        path: tuple[str, ...],
        *,
        resource_name: str | None = None,
    ) -> str | None:
        if path:
            root = path[0]
            if root == "cell":
                return "world"
            if root in {"ambient_temperature", "pressure", "velocity", "gas"}:
                return "gas"
            if root == "optics":
                if len(path) >= 2 and path[1] in {"visible_illumination", "cell_dose"}:
                    return "world"
                if len(path) >= 2 and path[1] == "gas_dose":
                    return "gas"
        if resource_name is None:
            return None
        if resource_name in {
            "cell_core",
            "island_id",
            "entity_id",
            "placeholder_displaced_material",
            "collapse_delay_pending",
            "visible_illumination",
            "cell_optical_dose",
        }:
            return "world"
        if resource_name in {
            "ambient_temperature",
            "pressure_ping",
            "flow_velocity",
            "gas_concentration",
            "gas_optical_dose",
        }:
            return "gas"
        return None

    def _serialize_readback_source_descriptor(self, path: tuple[str, ...], value: Any) -> Any:
        if isinstance(value, np.ndarray):
            payload = {
                "source_type": "cpu_array",
                "dtype": str(value.dtype),
                "shape": [int(dimension) for dimension in value.shape],
            }
            coord_space = self._infer_readback_payload_coord_space(path)
            if coord_space is not None:
                payload["coord_space"] = coord_space
            return payload
        if isinstance(value, GPUBufferReadbackSource):
            payload = {
                "source_type": "buffer_window",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "chunk_size": int(value.chunk_size),
                "start": int(value.start),
                "step": int(value.step),
                "count": int(value.count),
            }
            if value.dst_step is not None:
                payload["dst_step"] = int(value.dst_step)
            coord_space = self._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
            if coord_space is not None:
                payload["coord_space"] = coord_space
            return payload
        if isinstance(value, GPUCellCoreWindowReadbackSource):
            world_origin_x, world_origin_y = self._buffer_to_world_position((value.origin_x, value.origin_y))
            payload = {
                "source_type": "cell_core_window",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "coord_space": "world",
                "buffer_origin": [int(value.origin_x), int(value.origin_y)],
                "world_origin": [int(world_origin_x), int(world_origin_y)],
                "cell_grid_width": int(value.cell_grid_width),
            }
            if value.dst_cell_grid_width is not None:
                payload["dst_cell_grid_width"] = int(value.dst_cell_grid_width)
            return payload
        if isinstance(value, GPUGasWindowReadbackSource):
            gas_world_x, gas_world_y = self._buffer_gas_to_world_position((value.origin_x, value.origin_y))
            payload = {
                "source_type": "gas_window",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "coord_space": "gas",
                "buffer_origin": [int(value.origin_x), int(value.origin_y)],
                "gas_origin": [int(gas_world_x), int(gas_world_y)],
                "gas_grid_size": [int(value.gas_grid_width), int(value.gas_grid_height)],
                "species_id": int(value.species_id),
            }
            if value.dst_step is not None:
                payload["dst_step"] = int(value.dst_step)
            return payload
        if isinstance(value, GPUTextureReadbackSource):
            coord_space = self._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
            payload = {
                "source_type": "texture_view",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "components": int(value.components),
                "viewport": [int(part) for part in value.viewport],
            }
            if value.dst_step is not None:
                payload["dst_step"] = int(value.dst_step)
            if coord_space is not None:
                payload["coord_space"] = coord_space
                if coord_space == "world":
                    world_origin_x, world_origin_y = self._buffer_to_world_position((value.viewport[0], value.viewport[1]))
                    payload["world_origin"] = [int(world_origin_x), int(world_origin_y)]
                elif coord_space == "gas":
                    gas_world_x, gas_world_y = self._buffer_gas_to_world_position((value.viewport[0], value.viewport[1]))
                    payload["gas_origin"] = [int(gas_world_x), int(gas_world_y)]
            return payload
        if isinstance(value, GPUSegmentedBufferReadbackSource):
            payload = {
                "source_type": "segmented_buffer_window",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "grid_width": int(value.grid_width),
                "base_offset": int(value.base_offset),
                "segments": [
                    {
                        "src": [int(segment.src_x), int(segment.src_y)],
                        "dst": [int(segment.dst_x), int(segment.dst_y)],
                        "size": [int(segment.width), int(segment.height)],
                    }
                    for segment in value.segments
                ],
            }
            coord_space = self._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
            if coord_space is not None:
                payload["coord_space"] = coord_space
            return payload
        if isinstance(value, GPUSegmentedCellCoreWindowReadbackSource):
            payload = {
                "source_type": "segmented_cell_core_window",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "coord_space": "world",
                "cell_grid_width": int(value.cell_grid_width),
                "segments": [
                    {
                        "src": [int(segment.src_x), int(segment.src_y)],
                        "dst": [int(segment.dst_x), int(segment.dst_y)],
                        "size": [int(segment.width), int(segment.height)],
                    }
                    for segment in value.segments
                ],
            }
            return payload
        if isinstance(value, GPUSegmentedTextureReadbackSource):
            coord_space = self._infer_readback_payload_coord_space(path, resource_name=value.resource_name)
            payload = {
                "source_type": "segmented_texture_view",
                "resource_name": str(value.resource_name),
                "dtype": str(np.dtype(value.dtype)),
                "shape": [int(dimension) for dimension in value.shape],
                "components": int(value.components),
                "segments": [
                    {
                        "src": [int(segment.src_x), int(segment.src_y)],
                        "dst": [int(segment.dst_x), int(segment.dst_y)],
                        "size": [int(segment.width), int(segment.height)],
                    }
                    for segment in value.segments
                ],
            }
            if coord_space is not None:
                payload["coord_space"] = coord_space
            return payload
        if isinstance(value, dict):
            return {
                str(key): self._serialize_readback_source_descriptor(path + (str(key),), child)
                for key, child in value.items()
            }
        return self._normalize_json_payload_value(value)

    def _serialize_readback_plan_for_request(self, request: ReadbackRequest) -> dict[str, Any]:
        payload = self._make_readback_payload(request)
        plan = self.bridge._plan_readback_payload(payload)
        return {
            "request": self.serialize_readback_request(request),
            "layout": self.bridge._serialize_readback_layout(plan.layout),
            "nbytes": int(plan.nbytes),
            "gpu_source_count": int(len(plan.gpu_sources)),
            "cpu_chunk_count": int(len(plan.cpu_chunks)),
            "payload": self._serialize_readback_source_descriptor((), payload),
        }

    def _serialize_readback_plans_for_requests(self, requests: list[ReadbackRequest]) -> list[dict[str, Any]]:
        return [self._serialize_readback_plan_for_request(request) for request in requests]

    def _serialize_observation_plan_for_target_request(
        self,
        target: ObservationTarget,
        request: ReadbackRequest,
    ) -> dict[str, Any]:
        return {
            "target": self.serialize_observation_target(target),
            **self._serialize_readback_plan_for_request(request),
        }

    def _build_observation_request_pairs(
        self,
        targets: list[ObservationTarget],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[tuple[ObservationTarget, ReadbackRequest]]:
        pairs: list[tuple[ObservationTarget, ReadbackRequest]] = []
        for target in targets:
            request = self._build_observation_request(target, resolved_targets)
            if request is not None:
                pairs.append((target, request))
        return pairs

    def serialize_readback_plan(
        self,
        center_x: int | None,
        center_y: int | None,
        width: int,
        height: int,
        channels: tuple[str, ...],
        *,
        request_id: int | None = None,
        observer_id: int | None = None,
        label: str | None = None,
        target_query_id: str | None = None,
        target_dx: int = 0,
        target_dy: int = 0,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request = self.preview_readback(
            center_x,
            center_y,
            width,
            height,
            channels,
            request_id=request_id,
            observer_id=observer_id,
            label=label,
            target_query_id=target_query_id,
            target_dx=target_dx,
            target_dy=target_dy,
            target_queries=target_queries,
        )
        return self._serialize_readback_plan_for_request(request)

    def serialize_observation_plan(
        self,
        target: ObservationTarget | dict[str, Any],
        *,
        request_id: int | None = None,
        target_queries: list[TargetQuery | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        target = self._coerce_observation_target(target)
        request = self.preview_observation(
            target,
            request_id=request_id,
            target_queries=target_queries,
        )
        return self._serialize_observation_plan_for_target_request(target, request)

    def serialize_world_command(self, command: WorldCommand) -> dict[str, Any]:
        public_command = self._public_world_command(command)
        return {"kind": public_command.kind, "payload": self._normalize_json_payload_value(public_command.payload)}

    @staticmethod
    def serialize_entity_placeholder_input(placeholder: EntityPlaceholder) -> dict[str, Any]:
        return {
            "entity_id": int(placeholder.entity_id),
            "x": int(placeholder.world_x) if placeholder.world_x is not None else int(placeholder.x),
            "y": int(placeholder.world_y) if placeholder.world_y is not None else int(placeholder.y),
            "width": int(placeholder.width),
            "height": int(placeholder.height),
        }

    @staticmethod
    def serialize_target_query_input(query: TargetQuery) -> dict[str, Any]:
        return {
            "query_id": query.query_id,
            "anchor_filters": list(query.anchor_filters),
            "source_entity_id": None if query.source_entity_id is None else int(query.source_entity_id),
            "source_x": None if query.source_x is None else int(query.source_x),
            "source_y": None if query.source_y is None else int(query.source_y),
            "anchor_entity_id": None if query.anchor_entity_id is None else int(query.anchor_entity_id),
            "direction": query.direction,
            "distance_cells": int(query.distance_cells),
            "distance_meters": None if query.distance_meters is None else float(query.distance_meters),
            "distance_hint": query.distance_hint,
            "require_empty": bool(query.require_empty),
            "search_radius": int(query.search_radius),
            "label": query.label,
        }

    @staticmethod
    def serialize_page_stripe_update(update: PageStripeUpdate) -> dict[str, Any]:
        return {
            "axis": str(update.axis),
            "world_start": int(update.world_start),
            "world_end": int(update.world_end),
            "buffer_start": int(update.buffer_start),
            "buffer_end": int(update.buffer_end),
            "kind": str(update.kind),
            "cross_world_start": 0 if update.cross_world_start is None else int(update.cross_world_start),
            "cross_world_end": 0 if update.cross_world_end is None else int(update.cross_world_end),
        }

    @staticmethod
    def serialize_page_store_key(key: StoredStripeKey) -> dict[str, Any]:
        return {
            "axis": str(key.axis),
            "world_start": int(key.world_start),
            "world_end": int(key.world_end),
            "cross_world_start": int(getattr(key, "cross_world_start", 0)),
            "cross_world_end": int(getattr(key, "cross_world_end", 0)),
        }

    @classmethod
    def serialize_page_stripe_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return cls._normalize_json_payload_value(payload)

    @staticmethod
    def serialize_change_intent_input(intent: ChangeIntent) -> dict[str, Any]:
        return {
            "intent_id": intent.intent_id,
            "target_query_id": intent.target_query_id,
            "center_x": None if intent.center_x is None else int(intent.center_x),
            "center_y": None if intent.center_y is None else int(intent.center_y),
            "target_dx": int(intent.target_dx),
            "target_dy": int(intent.target_dy),
            "radius": int(intent.radius),
            "material": intent.material,
            "temperature_delta": float(intent.temperature_delta),
            "velocity": None if intent.velocity is None else [float(intent.velocity[0]), float(intent.velocity[1])],
            "velocity_carrier": intent.velocity_carrier,
            "velocity_mode": intent.velocity_mode,
            "require_empty": bool(intent.require_empty),
            "fallback_mode": intent.fallback_mode,
            "fallback_radius": int(intent.fallback_radius),
            "potency": float(intent.potency),
            "stability": float(intent.stability),
            "label": intent.label,
        }

    @staticmethod
    def serialize_carrier_intent_input(intent: CarrierIntent) -> dict[str, Any]:
        return {
            "intent_id": intent.intent_id,
            "kind": intent.kind,
            "target_query_id": intent.target_query_id,
            "center_x": None if intent.center_x is None else int(intent.center_x),
            "center_y": None if intent.center_y is None else int(intent.center_y),
            "source_entity_id": None if intent.source_entity_id is None else int(intent.source_entity_id),
            "source_x": None if intent.source_x is None else int(intent.source_x),
            "source_y": None if intent.source_y is None else int(intent.source_y),
            "target_dx": int(intent.target_dx),
            "target_dy": int(intent.target_dy),
            "radius": int(intent.radius),
            "material": intent.material,
            "gas_species": intent.gas_species,
            "gas_amount": float(intent.gas_amount),
            "light_type": intent.light_type,
            "light_strength": float(intent.light_strength),
            "light_spread": float(intent.light_spread),
            "force_radius": float(intent.force_radius),
            "force_strength": float(intent.force_strength),
            "force_lifetime": float(intent.force_lifetime),
            "release_mode": intent.release_mode,
            "require_empty": bool(intent.require_empty),
            "fallback_mode": intent.fallback_mode,
            "fallback_radius": int(intent.fallback_radius),
            "potency": float(intent.potency),
            "stability": float(intent.stability),
            "label": intent.label,
        }

    def serialize_frame_input(self, frame_input: WorldFrameInput) -> dict[str, Any]:
        return {
            "submission_id": frame_input.submission_id,
            "focus_center": None if frame_input.focus_center is None else list(frame_input.focus_center),
            "controller_state": deepcopy(frame_input.controller_state),
            "controller_state_provided": bool(frame_input.controller_state_provided),
            "entities": [self.serialize_entity_state_input(entity) for entity in frame_input.entities]
            if frame_input.entities is not None
            else None,
            "entity_placeholders": [self.serialize_entity_placeholder_input(placeholder) for placeholder in frame_input.entity_placeholders]
            if frame_input.entity_placeholders is not None
            else None,
            "force_sources": None
            if frame_input.force_sources is None
            else [
                {
                    "x": float(force_source.x),
                    "y": float(force_source.y),
                    "direction": [float(force_source.direction[0]), float(force_source.direction[1])],
                    "radius": float(force_source.radius),
                    "strength": float(force_source.strength),
                    "lifetime": float(force_source.lifetime),
                }
                for force_source in frame_input.force_sources
            ],
            "emitters": None
            if frame_input.emitters is None
            else [self._serialize_emitter_record(emitter) for emitter in frame_input.emitters],
            "target_queries": [self.serialize_target_query_input(query) for query in frame_input.target_queries],
            "change_intents": [self.serialize_change_intent_input(intent) for intent in frame_input.change_intents],
            "carrier_intents": [self.serialize_carrier_intent_input(intent) for intent in frame_input.carrier_intents],
            "observation_targets": [self.serialize_observation_target(target) for target in frame_input.observation_targets],
            "readback_requests": [self.serialize_readback_request(request) for request in frame_input.readback_requests],
            "commands": [self.serialize_world_command(command) for command in frame_input.commands],
        }

    def _serialize_readback_payload(self, payload: Any) -> Any:
        return self._normalize_json_payload_value(payload)

    def serialize_readback_result(self, result: ReadbackResult) -> dict[str, Any]:
        return {
            "frame_id": int(result.frame_id),
            "request": self.serialize_readback_request(result.request),
            "payload": self._serialize_readback_payload(result.payload),
        }

    def serialize_resolved_target(self, target: ResolvedTarget) -> dict[str, Any]:
        target = self._public_resolved_target(target)
        return {
            "query_id": target.query_id,
            "status": target.status,
            "anchor_filters": list(target.anchor_filters),
            "direction": target.direction,
            "distance_cells": int(target.distance_cells),
            "distance_meters": None if target.distance_meters is None else float(target.distance_meters),
            "distance_hint": target.distance_hint,
            "label": target.label,
            "source_position": None if target.source_position is None else list(target.source_position),
            "source_world_position": None
            if target.source_world_position is None
            else list(target.source_world_position),
            "anchor_kind": target.anchor_kind,
            "anchor_entity_id": target.anchor_entity_id,
            "anchor_position": None if target.anchor_position is None else list(target.anchor_position),
            "anchor_world_position": None
            if target.anchor_world_position is None
            else list(target.anchor_world_position),
            "resolved_position": None if target.resolved_position is None else list(target.resolved_position),
            "resolved_world_position": None
            if target.resolved_world_position is None
            else list(target.resolved_world_position),
            "note": target.note,
        }

    def serialize_resolved_change_intent(self, intent: ResolvedChangeIntent) -> dict[str, Any]:
        return {
            "intent_id": intent.intent_id,
            "status": intent.status,
            "target_query_id": intent.target_query_id,
            "label": intent.label,
            "potency": float(intent.potency),
            "stability": float(intent.stability),
            "center_position": None if intent.center_position is None else list(intent.center_position),
            "center_world_position": None
            if intent.center_world_position is None
            else list(intent.center_world_position),
            "effective_radius": int(intent.effective_radius),
            "material": intent.material,
            "temperature_delta": float(intent.temperature_delta),
            "velocity": None if intent.velocity is None else [float(intent.velocity[0]), float(intent.velocity[1])],
            "velocity_carrier": intent.velocity_carrier,
            "velocity_mode": intent.velocity_mode,
            "require_empty": bool(intent.require_empty),
            "fallback_mode": intent.fallback_mode,
            "fallback_applied": bool(intent.fallback_applied),
            "effect_shape": intent.effect_shape,
            "effect_cells": [list(cell) for cell in intent.effect_cells],
            "effect_bounds": None if intent.effect_bounds is None else list(intent.effect_bounds),
            "generated_commands": [self.serialize_world_command(command) for command in intent.generated_commands],
            "note": intent.note,
        }

    def serialize_resolved_carrier_intent(self, intent: ResolvedCarrierIntent) -> dict[str, Any]:
        return {
            "intent_id": intent.intent_id,
            "status": intent.status,
            "kind": intent.kind,
            "target_query_id": intent.target_query_id,
            "label": intent.label,
            "release_mode": intent.release_mode,
            "potency": float(intent.potency),
            "stability": float(intent.stability),
            "source_position": None if intent.source_position is None else list(intent.source_position),
            "source_world_position": None
            if intent.source_world_position is None
            else list(intent.source_world_position),
            "impact_position": None if intent.impact_position is None else list(intent.impact_position),
            "impact_world_position": None
            if intent.impact_world_position is None
            else list(intent.impact_world_position),
            "effective_radius": int(intent.effective_radius),
            "material": intent.material,
            "gas_species": intent.gas_species,
            "gas_amount": float(intent.gas_amount),
            "light_type": intent.light_type,
            "light_strength": float(intent.light_strength),
            "light_spread": float(intent.light_spread),
            "force_radius": float(intent.force_radius),
            "force_strength": float(intent.force_strength),
            "force_lifetime": float(intent.force_lifetime),
            "direction": None if intent.direction is None else [float(intent.direction[0]), float(intent.direction[1])],
            "require_empty": bool(intent.require_empty),
            "fallback_mode": intent.fallback_mode,
            "fallback_applied": bool(intent.fallback_applied),
            "effect_shape": intent.effect_shape,
            "effect_cells": [list(cell) for cell in intent.effect_cells],
            "effect_bounds": None if intent.effect_bounds is None else list(intent.effect_bounds),
            "generated_commands": [self.serialize_world_command(command) for command in intent.generated_commands],
            "note": intent.note,
        }

    def serialize_observation_result(self, result: ObservationResult) -> dict[str, Any]:
        return {
            "observer_id": int(result.observer_id),
            "frame_id": int(result.frame_id),
            "request": self.serialize_readback_request(result.request),
            "payload": self._serialize_readback_payload(result.payload),
        }

    @staticmethod
    def serialize_entity_observation_spec(spec: EntityObservationSpec) -> dict[str, Any]:
        return {
            "entity_id": int(spec.entity_id),
            "observe_channels": list(spec.observe_channels),
            "observe_pad_cells": int(spec.observe_pad_cells),
            "observe_width": None if spec.observe_width is None else int(spec.observe_width),
            "observe_height": None if spec.observe_height is None else int(spec.observe_height),
            "observe_label": spec.observe_label,
        }

    def serialize_entity_state_patch(self, patch: EntityStatePatch) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name, value in patch.fields.items():
            if name in ENTITY_STATE_PATCH_METADATA_FIELDS:
                continue
            if name == "x":
                fields[name] = int(patch.fields.get("_world_x", value))
            elif name == "y":
                fields[name] = int(patch.fields.get("_world_y", value))
            elif name in {"velocity_xy", "facing_xy"}:
                fields[name] = None if value is None else [float(item) for item in value]
            elif name == "tags":
                fields[name] = list(value)
            elif name == "observe_channels":
                fields[name] = list(value)
            else:
                fields[name] = value
        return {
            "entity_id": int(patch.entity_id),
            "fields": fields,
        }

    @staticmethod
    def serialize_observation_target(target: ObservationTarget) -> dict[str, Any]:
        return {
            "observer_id": int(target.observer_id),
            "channels": list(target.channels),
            "center_x": None if target.center_x is None else int(target.center_x),
            "center_y": None if target.center_y is None else int(target.center_y),
            "width": None if target.width is None else int(target.width),
            "height": None if target.height is None else int(target.height),
            "entity_id": None if target.entity_id is None else int(target.entity_id),
            "pad_cells": int(target.pad_cells),
            "label": target.label,
            "target_query_id": target.target_query_id,
            "target_dx": int(target.target_dx),
            "target_dy": int(target.target_dy),
        }

    @staticmethod
    def serialize_entity_state_input(entity: EntityState) -> dict[str, Any]:
        return {
            "entity_id": int(entity.entity_id),
            "x": int(entity.world_x) if entity.world_x is not None else int(entity.x),
            "y": int(entity.world_y) if entity.world_y is not None else int(entity.y),
            "width": int(entity.width),
            "height": int(entity.height),
            "velocity_xy": [float(entity.velocity_xy[0]), float(entity.velocity_xy[1])],
            "facing_xy": None if entity.facing_xy is None else [float(entity.facing_xy[0]), float(entity.facing_xy[1])],
            "placeholder_material": str(entity.placeholder_material),
            "tags": list(entity.tags),
            "observe_channels": list(entity.observe_channels),
            "observe_pad_cells": int(entity.observe_pad_cells),
            "observe_width": None if entity.observe_width is None else int(entity.observe_width),
            "observe_height": None if entity.observe_height is None else int(entity.observe_height),
            "observe_label": entity.observe_label,
        }

    def serialize_entity_state(self, entity: EntityState) -> dict[str, Any]:
        if entity.world_x is not None and entity.world_y is not None:
            world_x = int(entity.world_x)
            world_y = int(entity.world_y)
        else:
            world_x, world_y = self._buffer_to_world_position((int(entity.x), int(entity.y)))
        payload = self.serialize_entity_state_input(entity)
        payload["x"] = int(world_x)
        payload["y"] = int(world_y)
        return payload

    def serialize_entity_states(self) -> dict[str, Any]:
        entities = [self.serialize_entity_state(entity) for entity in sorted(self.entity_states.values(), key=lambda item: item.entity_id)]
        return {"entities": entities}

    def serialize_entity_observation_state(self) -> dict[str, Any]:
        entities = [entity for _, entity in sorted(self.entity_states.items())]
        _, targets = self._frame_entities_to_placeholders_and_observations(entities)
        requests = self._build_observation_requests(targets, {})
        return {
            "observations": [
                self.serialize_entity_observation_spec(
                    EntityObservationSpec(
                        entity_id=entity.entity_id,
                        observe_channels=entity.observe_channels,
                        observe_pad_cells=entity.observe_pad_cells,
                        observe_width=entity.observe_width,
                        observe_height=entity.observe_height,
                        observe_label=entity.observe_label,
                    )
                )
                for entity in entities
                if entity.observe_channels
            ],
            "targets": [self.serialize_observation_target(target) for target in targets],
            "requests": [self.serialize_readback_request(request) for request in requests],
        }

    def _current_cell_state_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
        if (
            self.simulation_backend == "gpu"
            and "cell_core" in self.bridge.gpu_authoritative_resources
            and self.bridge.enabled
            and self.bridge.ctx is not None
            and "cell_core" in self.bridge.buffers
        ):
            if not allow_gpu_sync_readback:
                return {
                    "material_id": self.material_id,
                    "phase": self.phase,
                    "integrity": self.integrity,
                }
            try:
                core = np.frombuffer(
                    self.bridge.buffers["cell_core"].read(size=self.width * self.height * 5 * np.dtype(np.uint32).itemsize),
                    dtype=np.uint32,
                ).reshape((self.height, self.width, 5))
                unpacked = unpack_cell_core(core)
                return {
                    "material_id": unpacked["material_id"].astype(np.int32, copy=False),
                    "phase": unpacked["phase"].astype(np.uint8, copy=False),
                    "integrity": unpacked["integrity"].astype(np.float32, copy=False),
                }
            except Exception as exc:
                raise RuntimeError(
                    "GPU-authoritative cell state is not directly readable from this thread; "
                    "use async readback for CPU-visible world snapshots"
                ) from exc
        return {
            "material_id": self.material_id,
            "phase": self.phase,
            "integrity": self.integrity,
        }

    def _current_entity_runtime_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, np.ndarray]:
        if (
            self.simulation_backend == "gpu"
            and self.bridge.enabled
            and self.bridge.ctx is not None
            and "entity_id" in self.bridge.gpu_authoritative_resources
            and "placeholder_displaced_material" in self.bridge.gpu_authoritative_resources
            and "entity_id" in self.bridge.buffers
            and "placeholder_displaced_material" in self.bridge.buffers
        ):
            if not allow_gpu_sync_readback:
                return {
                    "entity_id": self.entity_id,
                    "placeholder_displaced_material": self.placeholder_displaced_material,
                }
            try:
                return {
                    "entity_id": np.frombuffer(
                        self.bridge.buffers["entity_id"].read(size=self.entity_id.nbytes),
                        dtype=np.int32,
                    ).reshape(self.entity_id.shape),
                    "placeholder_displaced_material": np.frombuffer(
                        self.bridge.buffers["placeholder_displaced_material"].read(size=self.placeholder_displaced_material.nbytes),
                        dtype=np.int32,
                    ).reshape(self.placeholder_displaced_material.shape),
                }
            except Exception as exc:
                raise RuntimeError(
                    "GPU-authoritative entity runtime state is not directly readable from this thread; "
                    "use async readback for CPU-visible world snapshots"
                ) from exc
        return {
            "entity_id": self.entity_id,
            "placeholder_displaced_material": self.placeholder_displaced_material,
        }

    def _entity_placeholder_state_gpu_authoritative(self) -> bool:
        if self.simulation_backend != "gpu":
            return False
        authoritative = self.bridge.gpu_authoritative_resources
        return bool(
            "cell_core" in authoritative
            or "entity_id" in authoritative
            or "placeholder_displaced_material" in authoritative
        )

    def serialize_entity_placeholders(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
        if not allow_gpu_sync_readback and self._entity_placeholder_state_gpu_authoritative():
            return self.serialize_entity_placeholder_index_snapshot()
        payload: list[dict[str, Any]] = []
        cell_state = self._current_cell_state_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback)
        entity_runtime = self._current_entity_runtime_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback)
        material_id_grid = cell_state["material_id"]
        phase_grid = cell_state["phase"]
        displaced_grid = entity_runtime["placeholder_displaced_material"]
        for entity_id in sorted(self.entity_placeholders):
            cells = sorted(self.entity_placeholders[entity_id], key=lambda cell: (cell[1], cell[0]))
            if not cells:
                continue
            world_cells: list[tuple[int, int, int, int]] = []
            for buffer_x, buffer_y in cells:
                world_x, world_y = self._buffer_to_world_position((buffer_x, buffer_y))
                world_cells.append((int(world_x), int(world_y), int(buffer_x), int(buffer_y)))
            world_cells.sort(key=lambda cell: (cell[1], cell[0]))
            xs = [cell[0] for cell in world_cells]
            ys = [cell[1] for cell in world_cells]
            payload.append(
                {
                    "entity_id": int(entity_id),
                    "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                    "cells": [
                        {
                            "x": int(world_x),
                            "y": int(world_y),
                            "material_id": int(material_id_grid[buffer_y, buffer_x]),
                            "material": self._shadow_material_name(int(material_id_grid[buffer_y, buffer_x])),
                            "phase": int(phase_grid[buffer_y, buffer_x]),
                            "displaced_material_id": int(displaced_grid[buffer_y, buffer_x]),
                            "displaced_material": (
                                self._shadow_material_name(int(displaced_grid[buffer_y, buffer_x]))
                                if int(displaced_grid[buffer_y, buffer_x]) > 0
                                else None
                            ),
                        }
                        for world_x, world_y, buffer_x, buffer_y in world_cells
                    ],
                }
            )
        return {"placeholders": payload}

    def serialize_entity_placeholder_index_snapshot(self) -> dict[str, Any]:
        payload: list[dict[str, Any]] = []
        for entity_id in sorted(self.entity_placeholders):
            cells = sorted(self.entity_placeholders[entity_id], key=lambda cell: (cell[1], cell[0]))
            if not cells:
                continue
            entity = self.entity_states.get(int(entity_id))
            material_name = str(entity.placeholder_material) if entity is not None else "placeholder_solid"
            material_id = self._resolve_sanctioned_placeholder_material_id(material_name)
            if material_id <= 0:
                material_id = int(self.placeholder_material_id)
            world_cells: list[tuple[int, int]] = []
            for buffer_x, buffer_y in cells:
                world_x, world_y = self._buffer_to_world_position((buffer_x, buffer_y))
                world_cells.append((int(world_x), int(world_y)))
            world_cells.sort(key=lambda cell: (cell[1], cell[0]))
            xs = [cell[0] for cell in world_cells]
            ys = [cell[1] for cell in world_cells]
            payload.append(
                {
                    "entity_id": int(entity_id),
                    "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                    "cells": [
                        {
                            "x": int(world_x),
                            "y": int(world_y),
                            "material_id": int(material_id),
                            "material": self._shadow_material_name(int(material_id)),
                            "phase": int(Phase.STATIC_SOLID),
                            "displaced_material_id": 0,
                            "displaced_material": None,
                        }
                        for world_x, world_y in world_cells
                    ],
                }
            )
        return {"placeholders": payload}

    def serialize_entity_feedback_snapshot(self, *, allow_gpu_sync_readback: bool = False) -> dict[str, Any]:
        if not allow_gpu_sync_readback and self._entity_placeholder_state_gpu_authoritative():
            return self.serialize_consumed_entity_feedback_snapshot()
        feedback = {}
        for entity_id, entity in sorted(self.entity_states.items()):
            snapshot = self._build_entity_feedback_from_current_state(
                entity,
                allow_gpu_sync_readback=allow_gpu_sync_readback,
            )
            if snapshot is None:
                continue
            feedback[str(entity_id)] = self.serialize_entity_feedback(snapshot)
        return {"feedback": feedback}

    def serialize_consumed_entity_feedback_snapshot(self) -> dict[str, Any]:
        feedback = self.last_entity_observation_consume_snapshot.get("entity_feedback", {})
        if isinstance(feedback, dict):
            return {"feedback": deepcopy(feedback)}
        return {"feedback": {}}

    def _serialize_cpu_visible_entity_placeholders(self) -> dict[str, Any]:
        if self.simulation_backend == "gpu":
            return self.serialize_entity_placeholder_index_snapshot()
        return self.serialize_entity_placeholders()

    def serialize_entity_feedback(self, feedback: EntityFeedback) -> dict[str, Any]:
        return {
            "entity_id": int(feedback.entity_id),
            "bbox": list(feedback.bbox),
            "cells": [
                {
                    "x": int(cell.x),
                    "y": int(cell.y),
                    "present": bool(cell.present),
                    "material_id": int(cell.material_id),
                    "phase": int(cell.phase),
                    "integrity": float(cell.integrity),
                    "entity_id": int(cell.entity_id),
                }
                for cell in feedback.cells
            ],
        }

    def _store_entity_observation_consume_snapshot(
        self,
        *,
        frame_id: int,
        consumed_readbacks: list[ReadbackResult],
        observations: dict[int, ObservationResult],
        entity_feedback: dict[int, EntityFeedback],
    ) -> dict[str, Any]:
        snapshot = {
            "frame_id": int(frame_id),
            "consumed": len(consumed_readbacks),
            "consumed_readbacks": [self.serialize_readback_result(result) for result in consumed_readbacks],
            "observations": {
                str(observer_id): self.serialize_observation_result(result)
                for observer_id, result in observations.items()
            },
            "entity_feedback": {
                str(entity_id): self.serialize_entity_feedback(feedback)
                for entity_id, feedback in entity_feedback.items()
            },
        }
        self.last_entity_observation_consume_snapshot = snapshot
        return deepcopy(snapshot)

    def serialize_entity_observation_consume_state(self) -> dict[str, Any]:
        return deepcopy(self.last_entity_observation_consume_snapshot)

    def serialize_frame_output(self, output: WorldFrameOutput) -> dict[str, Any]:
        return {
            "frame_id": int(output.frame_id),
            "submission_id": output.submission_id,
            "controller_state": deepcopy(output.controller_state),
            "consumed_readbacks": [self.serialize_readback_result(result) for result in output.consumed_readbacks],
            "resolved_targets": {
                query_id: self.serialize_resolved_target(target)
                for query_id, target in output.resolved_targets.items()
            },
            "resolved_change_intents": {
                intent_id: self.serialize_resolved_change_intent(intent)
                for intent_id, intent in output.resolved_change_intents.items()
            },
            "resolved_carrier_intents": {
                intent_id: self.serialize_resolved_carrier_intent(intent)
                for intent_id, intent in output.resolved_carrier_intents.items()
            },
            "observations": {
                str(observer_id): self.serialize_observation_result(result)
                for observer_id, result in output.observations.items()
            },
            "entity_feedback": {
                str(entity_id): self.serialize_entity_feedback(feedback)
                for entity_id, feedback in output.entity_feedback.items()
            },
            "paging_updates": [asdict(update) for update in output.paging_updates],
            "observation_plans": [
                self._normalize_json_payload_value(plan)
                for plan in output.observation_plans
            ],
            "readback_plans": [
                self._normalize_json_payload_value(plan)
                for plan in output.readback_plans
            ],
            "bridge_upload_snapshot": self._normalize_json_payload_value(output.bridge_upload_snapshot),
            "bridge_frame_snapshot": self._normalize_json_payload_value(output.bridge_frame_snapshot),
            "queued_observations": int(output.queued_observations),
            "queued_readbacks": int(output.queued_readbacks),
            "queued_commands": int(output.queued_commands),
            "placeholder_count": int(output.placeholder_count),
        }

    def serialize_frame_preview(self, preview: WorldFramePreview) -> dict[str, Any]:
        return {
            "controller_state": deepcopy(preview.controller_state),
            "resolved_targets": {
                query_id: self.serialize_resolved_target(target)
                for query_id, target in preview.resolved_targets.items()
            },
            "resolved_change_intents": {
                intent_id: self.serialize_resolved_change_intent(intent)
                for intent_id, intent in preview.resolved_change_intents.items()
            },
            "resolved_carrier_intents": {
                intent_id: self.serialize_resolved_carrier_intent(intent)
                for intent_id, intent in preview.resolved_carrier_intents.items()
            },
            "resolved_commands": [self.serialize_world_command(command) for command in preview.resolved_commands],
            "observation_requests": [self.serialize_readback_request(request) for request in preview.observation_requests],
            "observation_plans": [
                self._normalize_json_payload_value(plan)
                for plan in preview.observation_plans
            ],
            "readback_requests": [self.serialize_readback_request(request) for request in preview.readback_requests],
            "readback_plans": [
                self._normalize_json_payload_value(plan)
                for plan in preview.readback_plans
            ],
            "bridge_frame_snapshot": self._normalize_json_payload_value(preview.bridge_frame_snapshot),
            "paging_updates": [asdict(update) for update in preview.paging_updates],
            "placeholder_count": int(preview.placeholder_count),
        }

    def serialize_debug_frame(
        self,
        view: DebugView | str,
        *,
        gas_species: str | None = None,
        light_type: str | None = None,
    ) -> dict[str, Any]:
        resolved_view = view if isinstance(view, DebugView) else DebugView(str(view).lower())
        if resolved_view == DebugView.GAS and gas_species is not None:
            if self._resolve_sanctioned_gas_id(str(gas_species)) < 0:
                raise KeyError(str(gas_species))
        if resolved_view in {DebugView.OPTICS, DebugView.LIGHT} and light_type is not None:
            if self._resolve_sanctioned_light_id(str(light_type)) < 0:
                raise KeyError(str(light_type))
        frame = self.debug_frame(
            resolved_view,
            gas_species=gas_species,
            light_type=light_type,
        )
        return {
            "view": resolved_view.value,
            "origin": [int(self.paging.origin_x), int(self.paging.origin_y)],
            "size": [int(self.width), int(self.height)],
            "gas_species": None if resolved_view != DebugView.GAS else str(gas_species or "water_gas"),
            "light_type": None if resolved_view not in {DebugView.OPTICS, DebugView.LIGHT} else light_type,
            "frame": np.asarray(frame, dtype=np.float32).round(4).tolist(),
        }

    def debug_frame(
        self,
        view: DebugView,
        *,
        gas_species: str | None = None,
        light_type: str | None = None,
    ) -> np.ndarray:
        if view == DebugView.MATERIAL:
            return self._material_frame()
        if view == DebugView.ACTIVE:
            return self._active_frame()
        if view == DebugView.TEMPERATURE:
            return self._temperature_frame()
        if view == DebugView.PRESSURE:
            return self._pressure_frame()
        if view == DebugView.HEAT:
            return self._heat_frame()
        if view == DebugView.LIQUID:
            return self._liquid_frame()
        if view == DebugView.REACTION:
            return self._reaction_frame()
        if view == DebugView.COLLAPSE:
            return self._collapse_frame()
        if view == DebugView.OPTICS:
            return self._optics_frame(light_type=light_type)
        if view == DebugView.VELOCITY:
            flow = self.sample_flow_to_cells()
            vectors = flow.copy()
            cell_speed = np.linalg.norm(self.velocity, axis=-1)
            use_cell_velocity = (self.material_id > 0) & (cell_speed > 1.0e-6)
            vectors[use_cell_velocity] = self.velocity[use_cell_velocity]
            return self._vector_field_frame(vectors)
        if view == DebugView.LIGHT:
            if light_type is not None:
                return self._optics_dose_frame(light_type=light_type)
            return np.clip(self.visible_illumination, 0.0, 1.0)
        if view == DebugView.MOTION:
            return self._motion_frame()
        return self._gas_frame(gas_species or "water_gas")

    def _material_frame(self) -> np.ndarray:
        frame = self.material_base_color[self.material_id]
        frame = frame * 0.35 + np.clip(self.visible_illumination, 0.0, 2.5)
        return np.clip(frame, 0.0, 1.0)

    def _temperature_frame(self) -> np.ndarray:
        temp = self.cell_temperature
        normalized = np.clip((temp - temp.min()) / max(1e-5, temp.max() - temp.min()), 0.0, 1.0)
        return np.stack([normalized, np.zeros_like(normalized), 1.0 - normalized], axis=-1)

    def _pressure_frame(self) -> np.ndarray:
        pressure = self.pressure_ping.astype(np.float32, copy=False)
        max_abs = float(
            max(
                1e-5,
                abs(float(pressure.min(initial=0.0))),
                abs(float(pressure.max(initial=0.0))),
            )
        )
        normalized = pressure / max_abs
        positive = np.clip(normalized, 0.0, 1.0)
        negative = np.clip(-normalized, 0.0, 1.0)
        magnitude = np.clip(np.abs(normalized), 0.0, 1.0)

        gas_frame = np.zeros((self.gas_height, self.gas_width, 3), dtype=np.float32)
        gas_frame[..., 0] = positive
        gas_frame[..., 1] = (1.0 - magnitude) * 0.18
        gas_frame[..., 2] = negative

        snapshot = self.gas_solver.runtime_snapshot()
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
        if solve_gas_mask.size > 0:
            gas_frame += solve_gas_mask[..., None] * np.array([0.03, 0.12, 0.03], dtype=np.float32)

        frame = np.repeat(
            np.repeat(gas_frame, self.gas_cell_size, axis=0),
            self.gas_cell_size,
            axis=1,
        )[: self.height, : self.width]
        return np.clip(frame, 0.0, 1.0)

    def _vector_field_frame(self, vectors: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(vectors, axis=-1)
        value = np.clip(magnitude / max(1e-5, float(magnitude.max(initial=0.0))), 0.0, 1.0)
        hue = (np.arctan2(vectors[..., 1], vectors[..., 0]) / (2.0 * np.pi) + 1.0) % 1.0
        hue6 = hue * 6.0
        sector = np.floor(hue6).astype(np.int32) % 6
        fraction = hue6 - np.floor(hue6)
        q = value * (1.0 - fraction)
        t = value * fraction
        rgb = np.zeros(vectors.shape[:-1] + (3,), dtype=np.float32)
        for index, components in (
            (0, (value, t, 0.0)),
            (1, (q, value, 0.0)),
            (2, (0.0, value, t)),
            (3, (0.0, q, value)),
            (4, (t, 0.0, value)),
            (5, (value, 0.0, q)),
        ):
            mask = sector == index
            if not np.any(mask):
                continue
            rgb[..., 0][mask] = components[0][mask] if isinstance(components[0], np.ndarray) else components[0]
            rgb[..., 1][mask] = components[1][mask] if isinstance(components[1], np.ndarray) else components[1]
            rgb[..., 2][mask] = components[2][mask] if isinstance(components[2], np.ndarray) else components[2]
        return rgb

    def _active_frame(self) -> np.ndarray:
        tile_ttl = np.asarray(self.active.active_tile_ttl, dtype=np.float32)
        active_chunk_mask = np.asarray(self.active.active_chunk_mask, dtype=np.float32)
        ttl_scale = max(1.0, float(self.active.active_ttl_reset))
        ttl_cells = np.repeat(
            np.repeat(tile_ttl / ttl_scale, self.active.tile_size, axis=0),
            self.active.tile_size,
            axis=1,
        )[: self.height, : self.width]
        chunk_span = self.active.tile_size * self.active.chunk_tiles
        chunk_cells = np.repeat(
            np.repeat(active_chunk_mask, chunk_span, axis=0),
            chunk_span,
            axis=1,
        )[: self.height, : self.width]
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        frame[..., 0] = ttl_cells * 0.10
        frame[..., 1] = ttl_cells * 0.95
        frame[..., 2] = chunk_cells * 0.35
        pending_mask = self.placeholder_displaced_material > 0
        if np.any(pending_mask):
            frame[pending_mask] = np.maximum(frame[pending_mask], np.array([0.95, 0.10, 0.95], dtype=np.float32))
        return np.clip(frame, 0.0, 1.0)

    def _motion_frame(self) -> np.ndarray:
        from oracle_game.sim.gpu_motion import (
            ISLAND_RESOLVE_BLOCKED,
            ISLAND_RESOLVE_DIRECT,
            ISLAND_RESOLVE_RERESOLVED,
            ISLAND_RESOLVE_STALE,
            POWDER_RESOLVE_BLOCKED,
            POWDER_RESOLVE_DDA,
            POWDER_RESOLVE_FALLBACK,
            POWDER_RESOLVE_STALE,
        )

        frame = np.clip(self._material_frame() * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
        snapshot = self.motion_solver.runtime_snapshot()

        powder_state_colors = {
            POWDER_RESOLVE_BLOCKED: np.array([1.0, 0.15, 0.15], dtype=np.float32),
            POWDER_RESOLVE_DDA: np.array([0.15, 1.0, 0.2], dtype=np.float32),
            POWDER_RESOLVE_FALLBACK: np.array([0.20, 0.85, 1.0], dtype=np.float32),
            POWDER_RESOLVE_STALE: np.array([1.0, 0.2, 0.8], dtype=np.float32),
        }
        island_state_colors = {
            ISLAND_RESOLVE_BLOCKED: np.array([1.0, 0.15, 0.15], dtype=np.float32),
            ISLAND_RESOLVE_DIRECT: np.array([0.15, 1.0, 0.2], dtype=np.float32),
            ISLAND_RESOLVE_RERESOLVED: np.array([0.20, 0.85, 1.0], dtype=np.float32),
            ISLAND_RESOLVE_STALE: np.array([1.0, 0.2, 0.8], dtype=np.float32),
        }

        for record in snapshot["powder_reservations"]:
            source_x, source_y = (int(value) for value in record["source_xy"])
            reserved_x, reserved_y = (int(value) for value in record["reserved_target_xy"])
            resolved_x, resolved_y = (int(value) for value in record["resolved_target_xy"])
            resolve_state = int(record["resolve_state"])
            self._accumulate_debug_point(frame, source_x, source_y, np.array([0.95, 0.35, 0.10], dtype=np.float32))
            if (reserved_x, reserved_y) != (source_x, source_y):
                self._accumulate_debug_point(frame, reserved_x, reserved_y, np.array([0.95, 0.80, 0.15], dtype=np.float32))
            self._accumulate_debug_point(
                frame,
                resolved_x,
                resolved_y,
                powder_state_colors.get(resolve_state, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            )

        for record in snapshot["island_reservations"]:
            x0, y0, x1, y1 = (int(value) for value in record["buffer_bbox"])
            target_dx, target_dy = (int(value) for value in record["target_shift"])
            resolved_dx, resolved_dy = (int(value) for value in record["resolved_shift"])
            resolve_state = int(record["resolve_state"])
            self._draw_debug_bbox_outline(frame, (x0, y0, x1, y1), np.array([0.85, 0.20, 0.95], dtype=np.float32))
            if target_dx != 0 or target_dy != 0:
                self._draw_debug_bbox_outline(
                    frame,
                    (x0 + target_dx, y0 + target_dy, x1 + target_dx, y1 + target_dy),
                    np.array([0.95, 0.80, 0.15], dtype=np.float32),
                )
            self._draw_debug_bbox_outline(
                frame,
                (x0 + resolved_dx, y0 + resolved_dy, x1 + resolved_dx, y1 + resolved_dy),
                island_state_colors.get(resolve_state, np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            )
        return np.clip(frame, 0.0, 1.0)

    def _heat_frame(self) -> np.ndarray:
        frame = np.clip(self._temperature_frame() * 0.45, 0.0, 1.0).astype(np.float32, copy=False)
        snapshot = self.heat_solver.runtime_snapshot()
        solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.float32)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
        phase_targets = np.asarray(snapshot["phase_targets"], dtype=np.int32)
        boil_targets = np.asarray(snapshot["boil_targets"], dtype=np.int32)
        condense_targets = np.asarray(snapshot["condense_targets"], dtype=np.bool_)

        if solve_cell_mask.size > 0:
            frame += solve_cell_mask[..., None] * np.array([0.02, 0.18, 0.02], dtype=np.float32)
        if solve_gas_mask.size > 0:
            solve_gas_cells = np.repeat(
                np.repeat(solve_gas_mask, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += solve_gas_cells[..., None] * np.array([0.08, 0.04, 0.18], dtype=np.float32)

        phase_ys, phase_xs = np.nonzero(phase_targets > 0)
        for y, x in zip(phase_ys.tolist(), phase_xs.tolist()):
            self._accumulate_debug_point(frame, int(x), int(y), np.array([0.95, 0.15, 0.95], dtype=np.float32))

        boil_ys, boil_xs = np.nonzero(boil_targets > 0)
        for y, x in zip(boil_ys.tolist(), boil_xs.tolist()):
            self._accumulate_debug_point(frame, int(x), int(y), np.array([1.0, 0.65, 0.05], dtype=np.float32))

        if condense_targets.size > 0:
            condense_any = np.any(condense_targets, axis=0).astype(np.float32, copy=False)
            condense_cells = np.repeat(
                np.repeat(condense_any, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += condense_cells[..., None] * np.array([0.14, 0.48, 0.62], dtype=np.float32)
        return np.clip(frame, 0.0, 1.0)

    def _liquid_frame(self) -> np.ndarray:
        frame = np.clip(self._material_frame() * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
        snapshot = self.liquid_solver.runtime_snapshot()
        post_cell_mask = np.asarray(snapshot["post_cell_mask"], dtype=np.float32)
        vertical_seam_mask = np.asarray(snapshot["vertical_seam_mask"], dtype=np.bool_)
        horizontal_seam_mask = np.asarray(snapshot["horizontal_seam_mask"], dtype=np.bool_)
        buoyancy_mask = np.asarray(snapshot["buoyancy_mask"], dtype=np.bool_)
        changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.bool_)

        if post_cell_mask.size > 0:
            frame += post_cell_mask[..., None] * np.array([0.02, 0.16, 0.06], dtype=np.float32)
        if np.any(changed_cell_mask):
            frame[changed_cell_mask] = np.maximum(frame[changed_cell_mask], np.array([0.18, 0.85, 1.0], dtype=np.float32))
        if np.any(vertical_seam_mask):
            frame[vertical_seam_mask] = np.maximum(frame[vertical_seam_mask], np.array([0.95, 0.18, 0.95], dtype=np.float32))
        if np.any(horizontal_seam_mask):
            frame[horizontal_seam_mask] = np.maximum(frame[horizontal_seam_mask], np.array([0.10, 0.92, 1.0], dtype=np.float32))
        if np.any(buoyancy_mask):
            frame[buoyancy_mask] = np.maximum(frame[buoyancy_mask], np.array([1.0, 0.78, 0.10], dtype=np.float32))
        pending_mask = self.placeholder_displaced_material > 0
        if np.any(pending_mask):
            frame[pending_mask] = np.maximum(frame[pending_mask], np.array([1.0, 0.18, 0.18], dtype=np.float32))
        return np.clip(frame, 0.0, 1.0)

    def _reaction_frame(self) -> np.ndarray:
        frame = np.clip(self._material_frame() * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
        snapshot = self.reaction_solver.runtime_snapshot()
        solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.float32)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
        changed_cell_mask = np.asarray(snapshot["changed_cell_mask"], dtype=np.bool_)
        changed_gas_mask = np.asarray(snapshot["changed_gas_mask"], dtype=np.float32)
        ambient_changed_mask = np.asarray(snapshot["ambient_changed_mask"], dtype=np.float32)
        timer_changed_mask = np.asarray(snapshot["timer_changed_mask"], dtype=np.bool_)
        emitted_light_mask = np.asarray(snapshot["emitted_light_mask"], dtype=np.bool_)
        emitted_material_mask = np.asarray(snapshot["emitted_material_mask"], dtype=np.bool_)

        if solve_cell_mask.size > 0:
            frame += solve_cell_mask[..., None] * np.array([0.02, 0.14, 0.02], dtype=np.float32)
        if solve_gas_mask.size > 0:
            solve_gas_cells = np.repeat(
                np.repeat(solve_gas_mask, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += solve_gas_cells[..., None] * np.array([0.04, 0.08, 0.18], dtype=np.float32)
        if np.any(changed_cell_mask):
            frame[changed_cell_mask] = np.maximum(frame[changed_cell_mask], np.array([0.18, 0.88, 1.0], dtype=np.float32))
        if ambient_changed_mask.size > 0:
            ambient_cells = np.repeat(
                np.repeat(ambient_changed_mask, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += ambient_cells[..., None] * np.array([0.92, 0.20, 0.92], dtype=np.float32)
        if changed_gas_mask.size > 0:
            gas_changed_cells = np.repeat(
                np.repeat(changed_gas_mask, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += gas_changed_cells[..., None] * np.array([0.12, 0.48, 0.92], dtype=np.float32)
        if np.any(timer_changed_mask):
            frame[timer_changed_mask] = np.maximum(frame[timer_changed_mask], np.array([1.0, 0.82, 0.16], dtype=np.float32))
        if np.any(emitted_light_mask):
            frame[emitted_light_mask] = np.maximum(frame[emitted_light_mask], np.array([1.0, 0.42, 0.08], dtype=np.float32))
        if np.any(emitted_material_mask):
            frame[emitted_material_mask] = np.maximum(frame[emitted_material_mask], np.array([1.0, 0.18, 0.85], dtype=np.float32))
        return np.clip(frame, 0.0, 1.0)

    def _collapse_frame(self) -> np.ndarray:
        frame = np.clip(self._material_frame() * 0.2, 0.0, 1.0).astype(np.float32, copy=False)
        snapshot = self.collapse_solver.runtime_snapshot(self)
        solve_region_mask = np.asarray(snapshot["solve_region_mask"], dtype=np.float32)
        support_seed_mask = np.asarray(snapshot["support_seed_mask"], dtype=np.bool_)
        supported_mask = np.asarray(snapshot["supported_mask"], dtype=np.bool_)
        unsupported_mask = np.asarray(snapshot["unsupported_mask"], dtype=np.bool_)
        delayed_pending_mask = np.asarray(snapshot["delayed_pending_mask"], dtype=np.bool_)
        immune_unsupported_mask = np.asarray(snapshot["immune_unsupported_mask"], dtype=np.bool_)
        collapsed_cell_mask = np.asarray(snapshot["collapsed_cell_mask"], dtype=np.bool_)

        if solve_region_mask.size > 0:
            frame += solve_region_mask[..., None] * np.array([0.04, 0.08, 0.18], dtype=np.float32)
        if np.any(supported_mask):
            frame[supported_mask] = np.maximum(frame[supported_mask], np.array([0.18, 0.82, 0.18], dtype=np.float32))
        if np.any(unsupported_mask):
            frame[unsupported_mask] = np.maximum(frame[unsupported_mask], np.array([1.0, 0.78, 0.10], dtype=np.float32))
        if np.any(delayed_pending_mask):
            frame[delayed_pending_mask] = np.maximum(frame[delayed_pending_mask], np.array([1.0, 0.46, 0.10], dtype=np.float32))
        if np.any(immune_unsupported_mask):
            frame[immune_unsupported_mask] = np.maximum(frame[immune_unsupported_mask], np.array([0.22, 0.78, 1.0], dtype=np.float32))
        if np.any(collapsed_cell_mask):
            frame[collapsed_cell_mask] = np.maximum(frame[collapsed_cell_mask], np.array([1.0, 0.20, 0.92], dtype=np.float32))
        if np.any(support_seed_mask):
            frame[support_seed_mask] = np.maximum(frame[support_seed_mask], np.array([0.16, 1.0, 1.0], dtype=np.float32))
        return np.clip(frame, 0.0, 1.0)

    def _optics_frame(self, *, light_type: str | None = None) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        snapshot = self.optics_solver.runtime_snapshot()
        solve_cell_mask = np.asarray(snapshot["solve_cell_mask"], dtype=np.float32)
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
        visible_changed_mask = np.asarray(snapshot["visible_changed_mask"], dtype=np.bool_)
        cell_dose_changed_mask = np.asarray(snapshot["cell_dose_changed_mask"], dtype=np.bool_)
        gas_dose_changed_mask = np.asarray(snapshot["gas_dose_changed_mask"], dtype=np.float32)
        emitter_origin_mask = np.asarray(snapshot["emitter_origin_mask"], dtype=np.bool_)
        dose_frame = self._optics_dose_frame(light_type=light_type)

        if solve_cell_mask.size > 0:
            frame += solve_cell_mask[..., None] * np.array([0.02, 0.12, 0.02], dtype=np.float32)
        if solve_gas_mask.size > 0:
            solve_gas_cells = np.repeat(
                np.repeat(solve_gas_mask, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += solve_gas_cells[..., None] * np.array([0.03, 0.05, 0.16], dtype=np.float32)
        frame = np.maximum(frame, dose_frame)
        if light_type is None and np.any(visible_changed_mask):
            frame[visible_changed_mask] = np.maximum(
                frame[visible_changed_mask],
                np.clip(self.visible_illumination[visible_changed_mask], 0.0, 1.0),
            )
        if light_type is None and np.any(cell_dose_changed_mask):
            frame[cell_dose_changed_mask] = np.maximum(frame[cell_dose_changed_mask], np.array([0.16, 0.95, 1.0], dtype=np.float32))
        if light_type is None and gas_dose_changed_mask.size > 0:
            gas_changed_cells = np.repeat(
                np.repeat(gas_dose_changed_mask, self.gas_cell_size, axis=0),
                self.gas_cell_size,
                axis=1,
            )[: self.height, : self.width]
            frame += gas_changed_cells[..., None] * np.array([0.14, 0.08, 0.42], dtype=np.float32)
        emitters = snapshot.get("emitters", [])
        if emitters:
            fallback_color = np.array([1.0, 0.42, 0.10], dtype=np.float32)
            for emitter in emitters:
                emitter_light = emitter.get("light_type")
                if light_type is not None and emitter_light != light_type:
                    continue
                origin = emitter.get("origin")
                if not isinstance(origin, tuple | list) or len(origin) != 2:
                    continue
                ox, oy = int(origin[0]), int(origin[1])
                light_id = self._resolve_sanctioned_light_id(str(emitter_light))
                if light_id < 0:
                    self._accumulate_debug_point(frame, ox, oy, fallback_color)
                    continue
                shadow_color = self._shadow_light_color(light_id)
                if shadow_color is None:
                    self._accumulate_debug_point(frame, ox, oy, fallback_color)
                    continue
                color = np.clip(shadow_color * 1.15, 0.0, 1.0)
                self._accumulate_debug_point(frame, ox, oy, color)
        elif np.any(emitter_origin_mask):
            frame[emitter_origin_mask] = np.maximum(frame[emitter_origin_mask], np.array([1.0, 0.42, 0.10], dtype=np.float32))
        return np.clip(frame, 0.0, 1.0)

    def _optics_dose_frame(self, *, light_type: str | None = None) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        if light_type is None:
            light_ids = [
                light_id
                for light_id in range(len(self.light_name_by_id))
                if self._shadow_light_name(light_id) is not None
            ]
        else:
            light_id = self._resolve_sanctioned_light_id(light_type)
            if light_id < 0:
                return frame
            light_ids = [light_id]
        for light_id in light_ids:
            dose_channel = self._shadow_light_dose_channel(light_id)
            color = self._shadow_light_color(light_id)
            if (
                dose_channel is None
                or color is None
                or dose_channel < 0
                or dose_channel >= self.cell_optical_dose.shape[0]
                or dose_channel >= self.gas_optical_dose.shape[0]
            ):
                continue
            cell_strength = 1.0 - np.exp(-np.clip(self.cell_optical_dose[dose_channel], 0.0, None))
            gas_strength = 1.0 - np.exp(
                -np.clip(
                    np.repeat(
                        np.repeat(self.gas_optical_dose[dose_channel], self.gas_cell_size, axis=0),
                        self.gas_cell_size,
                        axis=1,
                    )[: self.height, : self.width],
                    0.0,
                    None,
                )
                * 1.25
            )
            frame += color * cell_strength[..., None]
            frame += color * gas_strength[..., None] * 0.65
        return np.clip(frame, 0.0, 1.0)

    def _gas_frame(self, gas_species: str) -> np.ndarray:
        species_id = self._resolve_sanctioned_gas_id(gas_species)
        if species_id < 0:
            raise KeyError(gas_species)
        gas_field = self.gas_concentration[species_id]
        gas_cells = np.repeat(np.repeat(gas_field, self.gas_cell_size, axis=0), self.gas_cell_size, axis=1)[: self.height, : self.width]
        normalized = np.clip(gas_cells / max(1e-5, gas_cells.max(initial=1.0)), 0.0, 1.0)
        frame = np.stack([normalized * 0.3, normalized, normalized * 0.6], axis=-1).astype(np.float32, copy=False)
        snapshot = self.gas_solver.runtime_snapshot()
        solve_gas_mask = np.asarray(snapshot["solve_gas_mask"], dtype=np.float32)
        if solve_gas_mask.size > 0:
            solve_cells = np.repeat(np.repeat(solve_gas_mask, self.gas_cell_size, axis=0), self.gas_cell_size, axis=1)[: self.height, : self.width]
            frame += solve_cells[..., None] * np.array([0.25, 0.05, 0.35], dtype=np.float32)
        for force in self.force_sources:
            force_color = np.array([1.0, 0.45, 0.15], dtype=np.float32)
            center_x = int(round(force.x))
            center_y = int(round(force.y))
            self._accumulate_debug_point(frame, center_x, center_y, force_color)
            self._accumulate_debug_point(frame, center_x - 1, center_y, force_color)
            self._accumulate_debug_point(frame, center_x + 1, center_y, force_color)
            self._accumulate_debug_point(frame, center_x, center_y - 1, force_color)
            self._accumulate_debug_point(frame, center_x, center_y + 1, force_color)
        return np.clip(frame, 0.0, 1.0)

    def _accumulate_debug_point(self, frame: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            frame[y, x] = np.maximum(frame[y, x], color)

    def _draw_debug_bbox_outline(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        color: np.ndarray,
    ) -> None:
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(self.width, x0))
        y0 = max(0, min(self.height, y0))
        x1 = max(0, min(self.width, x1))
        y1 = max(0, min(self.height, y1))
        if x0 >= x1 or y0 >= y1:
            return
        frame[y0, x0:x1] = np.maximum(frame[y0, x0:x1], color)
        frame[y1 - 1, x0:x1] = np.maximum(frame[y1 - 1, x0:x1], color)
        frame[y0:y1, x0] = np.maximum(frame[y0:y1, x0], color)
        frame[y0:y1, x1 - 1] = np.maximum(frame[y0:y1, x1 - 1], color)

    def _apply_grid_world_commands(self, commands: list[WorldCommand]) -> None:
        if not commands:
            self.grid_command_pipeline.last_backend = "idle"
            return
        if self._gpu_pipeline_available(
            self.grid_command_pipeline,
            "world command",
            require=self.simulation_backend == "gpu",
        ):
            self.grid_command_pipeline.apply(self, commands)
            if self.simulation_backend == "gpu" and self._world_simulation_frame_active:
                active_rects: list[tuple[int, int, int, int, int]] = []
                for command in commands:
                    active_rect, collapse_rect = self._grid_world_command_runtime_regions(command)
                    if active_rect is not None:
                        active_rects.append(active_rect)
                    if collapse_rect is not None:
                        self._mark_collapse_dirty_rect(*collapse_rect)
                if active_rects and not self.bridge.mark_active_rects(self, active_rects):
                    self._require_gpu_stage("active scheduler command marking")
            else:
                for command in commands:
                    self._mark_grid_world_command_runtime_regions(command)
            if self.grid_command_pipeline.last_cpu_mirror_downloaded:
                self._rebuild_island_records()
            return
        self._require_cpu_oracle_backend("world command")
        self.grid_command_pipeline.last_backend = "cpu"
        for command in commands:
            self._apply_grid_world_command_cpu(command)

    def _apply_grid_world_command_cpu(self, command: WorldCommand) -> None:
        if command.kind == "inject_material":
            x, y = self._queued_command_xy(command)
            self._paint_material(x, y, command.payload["material"], command.payload["radius"])
        elif command.kind == "write_material_region":
            x, y = self._queued_command_xy(command)
            self._write_material_region_immediate(
                x,
                y,
                command.payload["width"],
                command.payload["height"],
                command.payload["material"],
            )
        elif command.kind == "inject_temperature":
            x, y = self._queued_command_xy(command)
            self._inject_temperature_immediate(x, y, command.payload["delta"], command.payload["radius"])
        elif command.kind == "inject_velocity":
            x, y = self._queued_command_xy(command)
            self._inject_velocity_immediate(
                x,
                y,
                tuple(command.payload["velocity"]),
                command.payload["radius"],
                carrier=command.payload.get("carrier", "cell"),
                mode=command.payload.get("mode", "add"),
            )
        elif command.kind == "inject_gas":
            x, y = self._queued_command_xy(command)
            self._inject_gas_immediate(x, y, command.payload["species"], command.payload["amount"], command.payload["radius"])

    def _grid_world_command_runtime_regions(
        self,
        command: WorldCommand,
    ) -> tuple[tuple[int, int, int, int, int] | None, tuple[int, int, int, int] | None]:
        x, y = self._queued_command_xy(command)
        if command.kind == "write_material_region":
            width = max(0, int(command.payload["width"]))
            height = max(0, int(command.payload["height"]))
            x0 = max(0, int(x) - 1)
            y0 = max(0, int(y) - 1)
            x1 = min(self.width, int(x) + width + 1)
            y1 = min(self.height, int(y) + height + 1)
            if x0 < x1 and y0 < y1:
                return (
                    (x0, y0, x1, y1, 0),
                    (
                        max(0, int(x)),
                        max(0, int(y)),
                        min(self.width, int(x) + width),
                        min(self.height, int(y) + height),
                    ),
                )
            return None, None
        radius = max(0, int(command.payload.get("radius", 0)))
        pad = 1 if command.kind == "inject_material" else 0
        active_rect = (
            max(0, int(x) - radius - pad),
            max(0, int(y) - radius - pad),
            min(self.width, int(x) + radius + pad + 1),
            min(self.height, int(y) + radius + pad + 1),
        )
        collapse_rect = None
        if command.kind == "inject_material":
            collapse_rect = (
                max(0, int(x) - radius),
                max(0, int(y) - radius),
                min(self.width, int(x) + radius + 1),
                min(self.height, int(y) + radius + 1),
            )
        return (*active_rect, 0), collapse_rect

    def _mark_grid_world_command_runtime_regions(self, command: WorldCommand) -> None:
        active_rect, collapse_rect = self._grid_world_command_runtime_regions(command)
        if active_rect is not None:
            x0, y0, x1, y1, tile_padding = active_rect
            self._mark_active_rect_runtime(x0, y0, x1, y1, tile_padding=tile_padding)
        if collapse_rect is not None:
            self._mark_collapse_dirty_rect(*collapse_rect)

    def _apply_commands(self) -> None:
        pending_grid_commands: list[WorldCommand] = []
        def flush_pending_grid_commands() -> None:
            if not pending_grid_commands:
                return
            if self.simulation_backend == "gpu" and not self._world_simulation_frame_active:
                self.bridge.sync_world(self)
            self._apply_grid_world_commands(pending_grid_commands)
            pending_grid_commands.clear()

        while self.command_queue:
            command = self.command_queue.popleft()
            self.bridge_frame_commands.append(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)))
            if command.kind in GPU_WORLD_COMMAND_KIND_IDS:
                if self.simulation_backend == "gpu":
                    self._gpu_pipeline_available(self.grid_command_pipeline, "world command")
                pending_grid_commands.append(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)))
                continue
            flush_pending_grid_commands()
            if command.kind == "inject_material":
                x, y = self._queued_command_xy(command)
                self._paint_material(x, y, command.payload["material"], command.payload["radius"])
            elif command.kind == "write_material_region":
                x, y = self._queued_command_xy(command)
                self._write_material_region_immediate(
                    x,
                    y,
                    command.payload["width"],
                    command.payload["height"],
                    command.payload["material"],
                )
            elif command.kind == "inject_temperature":
                x, y = self._queued_command_xy(command)
                self._inject_temperature_immediate(x, y, command.payload["delta"], command.payload["radius"])
            elif command.kind == "inject_velocity":
                x, y = self._queued_command_xy(command)
                self._inject_velocity_immediate(
                    x,
                    y,
                    tuple(command.payload["velocity"]),
                    command.payload["radius"],
                    carrier=command.payload.get("carrier", "cell"),
                    mode=command.payload.get("mode", "add"),
                )
            elif command.kind == "inject_force":
                x, y = self._queued_command_xy(command)
                self._append_force_source_immediate(
                    ForceSource(
                        x=float(x),
                        y=float(y),
                        direction=tuple(command.payload["direction"]),
                        radius=float(command.payload["radius"]),
                        strength=float(command.payload["strength"]),
                        lifetime=float(command.payload.get("lifetime", 0.5)),
                        world_x=float(command.payload["x"]),
                        world_y=float(command.payload["y"]),
                    )
                )
            elif command.kind == "inject_gas":
                x, y = self._queued_command_xy(command)
                self._inject_gas_immediate(x, y, command.payload["species"], command.payload["amount"], command.payload["radius"])
            elif command.kind == "inject_light":
                light_type = command.payload["light_type"]
                light_id = self._resolve_sanctioned_light_id(str(light_type))
                if light_id < 0:
                    continue
                if "radius" in command.payload:
                    range_cells = int(command.payload["radius"])
                else:
                    shadow_default_range = self._shadow_light_default_range(light_id)
                    if shadow_default_range is None:
                        continue
                    range_cells = int(shadow_default_range)
                x, y = self._queued_command_xy(command)
                shadow_light = self._shadow_light_name(light_id)
                if shadow_light is None:
                    continue
                self._append_transient_light_emitter_immediate(
                    {
                        "light_type": shadow_light,
                        "origin": (x, y),
                        "world_origin": (int(command.payload["x"]), int(command.payload["y"])),
                        "direction": tuple(command.payload.get("direction", (0.0, 0.0))),
                        "spread": float(command.payload.get("spread", 0.25)),
                        "strength": command.payload["strength"],
                        "range_cells": range_cells,
                    }
                )
            elif command.kind == "sync_entity_placeholders":
                payload = command.payload.get("placeholders", [])
                self._sync_entity_placeholders(
                    [
                        self._frame_entity_placeholder_input(placeholder)
                        if isinstance(placeholder, EntityPlaceholder)
                        else self._frame_entity_placeholder_input(EntityPlaceholder(**placeholder))
                        for placeholder in payload
                    ]
                )
            elif command.kind == "sync_entity_states":
                payload = command.payload.get("entities", [])
                entities = [
                    self._frame_entity_state_input(entity)
                    if isinstance(entity, EntityState)
                    else self._frame_entity_state_input(self._coerce_entity_state(entity))
                    for entity in payload
                ]
                placeholders, _ = self._sync_entity_states(entities)
                self._sync_entity_placeholders(placeholders)
            elif command.kind == "patch_entity_states":
                payload = command.payload.get("patches", [])
                self._patch_entity_states(
                    [
                        self._frame_entity_state_patch_input(patch)
                        if isinstance(patch, EntityStatePatch)
                        else self._frame_entity_state_patch_input(self._coerce_entity_state_patch(patch))
                        for patch in payload
                    ]
                )
            elif command.kind == "sync_entity_observation_specs":
                payload = command.payload.get("observations", [])
                self._sync_entity_observation_specs(
                    [
                        observation
                        if isinstance(observation, EntityObservationSpec)
                        else self._coerce_entity_observation_spec(observation)
                        for observation in payload
                    ]
                )
            elif command.kind == "set_force_sources":
                payload = command.payload.get("force_sources", [])
                self._sync_force_sources(
                    [
                        self._public_force_source_input(force_source)
                        for force_source in payload
                    ]
                )
            elif command.kind == "set_emitters":
                payload = command.payload.get("emitters", [])
                normalized_emitters = []
                for emitter in payload:
                    record = self._coerce_emitter(emitter)
                    normalized_emitters.append(
                        {
                            **dict(record),
                            "origin": self._world_to_buffer_clamped(
                                int(record["world_origin"][0]),
                                int(record["world_origin"][1]),
                            ),
                        }
                    )
                self._sync_persistent_emitters(normalized_emitters)
            elif command.kind == "advance_paging":
                self._advance_paging(command.payload["center_x"], command.payload["center_y"])
            elif command.kind == "apply_page_stripe":
                update_payload = command.payload["update"]
                update = update_payload if isinstance(update_payload, PageStripeUpdate) else PageStripeUpdate(**update_payload)
                self.bridge_frame_paging_updates.append(PageStripeUpdate(**asdict(update)))
                self._apply_page_stripe(update, command.payload["payload"])
                self._record_bridge_page_stripe(update, command.payload["payload"])
            elif command.kind == "reset_world":
                self._reset_world_state(reset_bridge_frame_inputs=True, keep_command_log=True)
            elif command.kind == "request_readback":
                request = self._assign_readback_request_id(self._normalize_readback_request(ReadbackRequest(**command.payload)))
                self.pending_readbacks.append(request)
                self.bridge_frame_readback_requests.append(replace(request))
            elif command.kind == "update_material_table":
                self.update_material_table(command.payload["materials"], immediate=True)
            elif command.kind == "update_gas_species_table":
                self.update_gas_species_table(command.payload["gases"], immediate=True)
            elif command.kind == "update_light_type_table":
                self.update_light_type_table(command.payload["lights"], immediate=True)
            elif command.kind == "update_material_optics_table":
                self.update_material_optics_table(command.payload["optics"], immediate=True)
            elif command.kind == "update_reaction_table":
                self.update_reaction_table(command.payload["actions"], command.payload["rules"], immediate=True)
            elif command.kind == "replace_reaction_table":
                self.replace_reaction_table(command.payload["actions"], command.payload["rules"], immediate=True)
            elif command.kind == "patch_material":
                self.patch_material(command.payload["name"], immediate=True, **command.payload["fields"])
            elif command.kind == "patch_light":
                self.patch_light(command.payload["name"], immediate=True, **command.payload["fields"])
            elif command.kind == "patch_gas":
                self.patch_gas(command.payload["name"], immediate=True, **command.payload["fields"])
            elif command.kind == "patch_material_optics":
                self.patch_material_optics(
                    command.payload["material_name"],
                    command.payload["light_type"],
                    immediate=True,
                    **command.payload["fields"],
                )
            elif command.kind == "patch_reaction_action":
                self.patch_reaction_action(command.payload["index"], immediate=True, **command.payload["fields"])
            elif command.kind == "delete_reaction_action":
                self.delete_reaction_action(command.payload["index"], immediate=True)
            elif command.kind == "patch_reaction_rule":
                self.patch_reaction_rule(
                    command.payload["rule_set"],
                    command.payload["index"],
                    immediate=True,
                    **command.payload["fields"],
                )
            elif command.kind == "delete_reaction_rule":
                self.delete_reaction_rule(
                    command.payload["rule_set"],
                    command.payload["index"],
                    immediate=True,
                )
        flush_pending_grid_commands()

    def _finish_readbacks(self, *, world_synced: bool = False) -> None:
        normalized_requests = [self._assign_readback_request_id(self._normalize_readback_request(request)) for request in self.pending_readbacks]
        self.pending_readbacks[:] = normalized_requests
        if self.pending_readbacks and not world_synced and self.simulation_backend != "gpu":
            self.bridge.sync_world(self)
        remaining_pending: list[ReadbackRequest] = []
        readback_upload_dirty = False
        for request in self.pending_readbacks:
            payload = self._make_readback_payload(request)
            if not self.bridge.queue_readback(
                self.frame_id,
                request,
                payload,
                require_gpu_sources=self.simulation_backend == "gpu",
            ):
                remaining_pending.append(request)
                continue
            readback_upload_dirty = True
            if request not in self.bridge_frame_readback_requests:
                self.bridge_frame_readback_requests.append(replace(request))
            if not any(existing.request_id == request.request_id for existing in self.inflight_readbacks):
                self.inflight_readbacks.append(replace(request))
        self.pending_readbacks[:] = remaining_pending
        if readback_upload_dirty:
            self.bridge.sync_readback_requests(self)

    def _collect_ready_readbacks(self, current_frame_id: int) -> None:
        while True:
            result = self.bridge.poll_readback(current_frame_id)
            if result is None:
                return
            if result.request.request_id is not None:
                self.inflight_readbacks = [
                    request for request in self.inflight_readbacks if request.request_id != result.request.request_id
                ]
                if int(result.request.request_id) in self.canceled_readback_request_ids:
                    continue
            self.completed_readbacks.append(result)

    def _queued_command_xy(self, command: WorldCommand) -> tuple[int, int]:
        x = int(command.payload["x"])
        y = int(command.payload["y"])
        if any(
            key in command.payload
            for key in ("resolved_target_query_id", "resolved_change_intent_id", "resolved_carrier_intent_id")
        ):
            return self._world_to_buffer_clamped(x, y)
        return self._world_to_buffer_clamped(x, y)

    def _make_readback_payload(self, request: ReadbackRequest) -> dict[str, Any]:
        world_x0, world_y0, world_x1, world_y1 = self._centered_world_window(
            int(request.center_x),
            int(request.center_y),
            int(request.width),
            int(request.height),
        )
        cell_width = world_x1 - world_x0
        cell_height = world_y1 - world_y0
        x_spans = self._world_axis_spans(world_x0, world_x1, axis="x")
        y_spans = self._world_axis_spans(world_y0, world_y1, axis="y")
        cell_contiguous = len(x_spans) <= 1 and len(y_spans) <= 1
        x0 = x_spans[0][0] if x_spans else 0
        y0 = y_spans[0][0] if y_spans else 0
        x1 = x0 + cell_width
        y1 = y0 + cell_height
        gas_world_x0, gas_world_y0, gas_world_x1, gas_world_y1 = self._world_gas_window_for_cell_world_rect(
            world_x0,
            world_y0,
            world_x1,
            world_y1,
        )
        gas_width = gas_world_x1 - gas_world_x0
        gas_height = gas_world_y1 - gas_world_y0
        gx_spans = self._world_axis_spans(gas_world_x0, gas_world_x1, axis="x", gas_grid=True)
        gy_spans = self._world_axis_spans(gas_world_y0, gas_world_y1, axis="y", gas_grid=True)
        gas_contiguous = len(gx_spans) <= 1 and len(gy_spans) <= 1
        gx0 = gx_spans[0][0] if gx_spans else 0
        gy0 = gy_spans[0][0] if gy_spans else 0
        gx1 = gx0 + gas_width
        gy1 = gy0 + gas_height
        gpu_mode = self.simulation_backend == "gpu"

        def axis_segments(spans: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
            dst_offset = 0
            result: list[tuple[int, int, int, int]] = []
            for src_start, src_end in spans:
                length = max(0, int(src_end) - int(src_start))
                if length <= 0:
                    continue
                result.append((int(src_start), int(src_end), dst_offset, length))
                dst_offset += length
            return result

        def rect_segments(
            x_axis_spans: list[tuple[int, int]],
            y_axis_spans: list[tuple[int, int]],
        ) -> tuple[GPUReadbackSegment, ...]:
            x_parts = axis_segments(x_axis_spans)
            y_parts = axis_segments(y_axis_spans)
            return tuple(
                GPUReadbackSegment(
                    src_x=src_x0,
                    src_y=src_y0,
                    dst_x=dst_x0,
                    dst_y=dst_y0,
                    width=width,
                    height=height,
                )
                for src_y0, _src_y1, dst_y0, height in y_parts
                for src_x0, _src_x1, dst_x0, width in x_parts
            )

        cell_segments = rect_segments(x_spans, y_spans)
        gas_segments = rect_segments(gx_spans, gy_spans)

        def buffer_window_source(
            *,
            resource_name: str,
            dtype: str | np.dtype[Any],
            shape: tuple[int, int],
            grid_width: int,
            origin_x: int,
            origin_y: int,
            contiguous: bool,
            segments: tuple[GPUReadbackSegment, ...],
            base_offset: int = 0,
            cpu_array_factory: Any,
        ) -> Any:
            if not gpu_mode:
                return cpu_array_factory()
            resolved_dtype = np.dtype(dtype)
            itemsize = resolved_dtype.itemsize
            if contiguous:
                row_bytes = int(grid_width) * itemsize
                window_bytes = int(shape[1]) * itemsize
                start = int(base_offset) + (int(origin_y) * int(grid_width) + int(origin_x)) * itemsize
                return GPUBufferReadbackSource(
                    resource_name=resource_name,
                    dtype=resolved_dtype.str,
                    shape=shape,
                    chunk_size=window_bytes,
                    start=start,
                    step=row_bytes,
                    count=int(shape[0]),
                )
            return GPUSegmentedBufferReadbackSource(
                resource_name=resource_name,
                dtype=resolved_dtype.str,
                shape=shape,
                grid_width=int(grid_width),
                base_offset=int(base_offset),
                segments=segments,
            )

        def texture_window_source(
            *,
            resource_name: str,
            dtype: str | np.dtype[Any],
            shape: tuple[int, ...],
            components: int,
            origin_x: int,
            origin_y: int,
            width: int,
            height: int,
            contiguous: bool,
            segments: tuple[GPUReadbackSegment, ...],
            cpu_array_factory: Any,
        ) -> Any:
            if not gpu_mode:
                return cpu_array_factory()
            if contiguous:
                return GPUTextureReadbackSource(
                    resource_name=resource_name,
                    dtype=np.dtype(dtype).str,
                    shape=shape,
                    components=int(components),
                    viewport=(int(origin_x), int(origin_y), int(width), int(height)),
                )
            return GPUSegmentedTextureReadbackSource(
                resource_name=resource_name,
                dtype=np.dtype(dtype).str,
                shape=shape,
                components=int(components),
                segments=segments,
            )

        def cell_core_source() -> Any:
            if not gpu_mode:
                return self._pack_cell_core_world_window(world_x0, world_y0, world_x1, world_y1)
            if cell_contiguous:
                return GPUCellCoreWindowReadbackSource(
                    resource_name="cell_core",
                    dtype="u4",
                    shape=(cell_height, cell_width, 5),
                    cell_grid_width=self.width,
                    origin_x=x0,
                    origin_y=y0,
                )
            return GPUSegmentedCellCoreWindowReadbackSource(
                resource_name="cell_core",
                dtype="u4",
                shape=(cell_height, cell_width, 5),
                cell_grid_width=self.width,
                segments=cell_segments,
            )

        def gas_species_source(species_id: int) -> Any:
            if not gpu_mode:
                return self._extract_world_window(
                    self.gas_concentration[species_id],
                    gas_world_x0,
                    gas_world_y0,
                    gas_world_x1,
                    gas_world_y1,
                    x_axis=1,
                    y_axis=0,
                    gas_grid=True,
                ).astype(np.float32, copy=False)
            if gas_contiguous:
                return GPUGasWindowReadbackSource(
                    resource_name="gas_concentration",
                    dtype="f4",
                    shape=(gas_height, gas_width),
                    gas_grid_width=self.gas_width,
                    gas_grid_height=self.gas_height,
                    species_id=int(species_id),
                    origin_x=gx0,
                    origin_y=gy0,
                )
            return GPUSegmentedBufferReadbackSource(
                resource_name="gas_concentration",
                dtype=np.dtype(np.float32).str,
                shape=(gas_height, gas_width),
                grid_width=self.gas_width,
                base_offset=int(species_id) * self.gas_height * self.gas_width * np.dtype(np.float32).itemsize,
                segments=gas_segments,
            )

        payload: dict[str, Any] = {}
        if "cell" in request.channels:
            cell_payload: dict[str, Any] = {
                "origin": [world_x0, world_y0],
                "size": [cell_width, cell_height],
            }
            cell_payload.update(
                {
                    "core_words": cell_core_source(),
                    "island_id": buffer_window_source(
                        resource_name="island_id",
                        dtype=np.int32,
                        shape=(cell_height, cell_width),
                        grid_width=self.width,
                        origin_x=x0,
                        origin_y=y0,
                        contiguous=cell_contiguous,
                        segments=cell_segments,
                        cpu_array_factory=lambda: self._extract_world_window(
                            self.island_id,
                            world_x0,
                            world_y0,
                            world_x1,
                            world_y1,
                            x_axis=1,
                            y_axis=0,
                        ).astype(np.int32, copy=False),
                    ),
                    "entity_id": buffer_window_source(
                        resource_name="entity_id",
                        dtype=np.int32,
                        shape=(cell_height, cell_width),
                        grid_width=self.width,
                        origin_x=x0,
                        origin_y=y0,
                        contiguous=cell_contiguous,
                        segments=cell_segments,
                        cpu_array_factory=lambda: self._extract_world_window(
                            self.entity_id,
                            world_x0,
                            world_y0,
                            world_x1,
                            world_y1,
                            x_axis=1,
                            y_axis=0,
                        ).astype(np.int32, copy=False),
                    ),
                    "placeholder_displaced_material": buffer_window_source(
                        resource_name="placeholder_displaced_material",
                        dtype=np.int32,
                        shape=(cell_height, cell_width),
                        grid_width=self.width,
                        origin_x=x0,
                        origin_y=y0,
                        contiguous=cell_contiguous,
                        segments=cell_segments,
                        cpu_array_factory=lambda: self._extract_world_window(
                            self.placeholder_displaced_material,
                            world_x0,
                            world_y0,
                            world_x1,
                            world_y1,
                            x_axis=1,
                            y_axis=0,
                        ).astype(np.int32, copy=False),
                    ),
                    "collapse_delay_pending": buffer_window_source(
                        resource_name="collapse_delay_pending",
                        dtype=np.int32,
                        shape=(cell_height, cell_width),
                        grid_width=self.width,
                        origin_x=x0,
                        origin_y=y0,
                        contiguous=cell_contiguous,
                        segments=cell_segments,
                        cpu_array_factory=lambda: self._extract_world_window(
                            self.collapse_delay_pending,
                            world_x0,
                            world_y0,
                            world_x1,
                            world_y1,
                            x_axis=1,
                            y_axis=0,
                        ).astype(np.int32, copy=False),
                    ),
                }
            )
            payload["cell"] = cell_payload
        if "ambient_temperature" in request.channels:
            payload["ambient_temperature"] = {
                "origin": [gas_world_x0, gas_world_y0],
                "size": [gas_width, gas_height],
                "grid": "gas",
                "values": texture_window_source(
                    resource_name="ambient_temperature",
                    dtype=np.float32,
                    shape=(gas_height, gas_width),
                    components=1,
                    origin_x=gx0,
                    origin_y=gy0,
                    width=gas_width,
                    height=gas_height,
                    contiguous=gas_contiguous,
                    segments=gas_segments,
                    cpu_array_factory=lambda: self._extract_world_window(
                        self.ambient_temperature,
                        gas_world_x0,
                        gas_world_y0,
                        gas_world_x1,
                        gas_world_y1,
                        x_axis=1,
                        y_axis=0,
                        gas_grid=True,
                    ).astype(np.float32, copy=False),
                ),
            }
        if "pressure" in request.channels:
            payload["pressure"] = {
                "origin": [gas_world_x0, gas_world_y0],
                "size": [gas_width, gas_height],
                "grid": "gas",
                "values": texture_window_source(
                    resource_name="pressure_ping",
                    dtype=np.float32,
                    shape=(gas_height, gas_width),
                    components=1,
                    origin_x=gx0,
                    origin_y=gy0,
                    width=gas_width,
                    height=gas_height,
                    contiguous=gas_contiguous,
                    segments=gas_segments,
                    cpu_array_factory=lambda: self._extract_world_window(
                        self.pressure_ping,
                        gas_world_x0,
                        gas_world_y0,
                        gas_world_x1,
                        gas_world_y1,
                        x_axis=1,
                        y_axis=0,
                        gas_grid=True,
                    ).astype(np.float32, copy=False),
                ),
            }
        if "velocity" in request.channels:
            payload["velocity"] = {
                "origin": [gas_world_x0, gas_world_y0],
                "size": [gas_width, gas_height],
                "grid": "gas",
                "values": texture_window_source(
                    resource_name="flow_velocity",
                    dtype=np.float32,
                    shape=(gas_height, gas_width, 2),
                    components=2,
                    origin_x=gx0,
                    origin_y=gy0,
                    width=gas_width,
                    height=gas_height,
                    contiguous=gas_contiguous,
                    segments=gas_segments,
                    cpu_array_factory=lambda: self._extract_world_window(
                        self.flow_velocity,
                        gas_world_x0,
                        gas_world_y0,
                        gas_world_x1,
                        gas_world_y1,
                        x_axis=1,
                        y_axis=0,
                        gas_grid=True,
                    ).astype(np.float32, copy=False),
                ),
            }
        if "optics" in request.channels:
            light_entries = [
                (shadow_name, dose_channel)
                for light_id in range(len(self.light_name_by_id))
                for shadow_name in [self._shadow_light_name(light_id)]
                for dose_channel in [self._shadow_light_dose_channel(light_id)]
                if shadow_name
                and dose_channel is not None
                and 0 <= int(dose_channel) < self.cell_optical_dose.shape[0]
                and 0 <= int(dose_channel) < self.gas_optical_dose.shape[0]
            ]
            optics_payload: dict[str, Any] = {
                "origin": [world_x0, world_y0],
                "size": [cell_width, cell_height],
                "gas_origin": [gas_world_x0, gas_world_y0],
                "gas_size": [gas_width, gas_height],
            }
            optics_payload["visible_illumination"] = texture_window_source(
                resource_name="visible_illumination",
                dtype=np.float32,
                shape=(cell_height, cell_width, 3),
                components=3,
                origin_x=x0,
                origin_y=y0,
                width=cell_width,
                height=cell_height,
                contiguous=cell_contiguous,
                segments=cell_segments,
                cpu_array_factory=lambda: self._extract_world_window(
                    self.visible_illumination,
                    world_x0,
                    world_y0,
                    world_x1,
                    world_y1,
                    x_axis=1,
                    y_axis=0,
                ).astype(np.float32, copy=False),
            )
            optics_payload["cell_dose"] = {
                light_name: buffer_window_source(
                    resource_name="cell_optical_dose",
                    dtype=np.float32,
                    shape=(cell_height, cell_width),
                    grid_width=self.width,
                    origin_x=x0,
                    origin_y=y0,
                    contiguous=cell_contiguous,
                    segments=cell_segments,
                    base_offset=int(dose_channel) * self.height * self.width * np.dtype(np.float32).itemsize,
                    cpu_array_factory=lambda dose_channel=dose_channel: self._extract_world_window(
                        self.cell_optical_dose[dose_channel],
                        world_x0,
                        world_y0,
                        world_x1,
                        world_y1,
                        x_axis=1,
                        y_axis=0,
                    ).astype(np.float32, copy=False),
                )
                for light_name, dose_channel in light_entries
            }
            optics_payload["gas_dose"] = {
                light_name: buffer_window_source(
                    resource_name="gas_optical_dose",
                    dtype=np.float32,
                    shape=(gas_height, gas_width),
                    grid_width=self.gas_width,
                    origin_x=gx0,
                    origin_y=gy0,
                    contiguous=gas_contiguous,
                    segments=gas_segments,
                    base_offset=int(dose_channel) * self.gas_height * self.gas_width * np.dtype(np.float32).itemsize,
                    cpu_array_factory=lambda dose_channel=dose_channel: self._extract_world_window(
                        self.gas_optical_dose[dose_channel],
                        gas_world_x0,
                        gas_world_y0,
                        gas_world_x1,
                        gas_world_y1,
                        x_axis=1,
                        y_axis=0,
                        gas_grid=True,
                    ).astype(np.float32, copy=False),
                )
                for light_name, dose_channel in light_entries
            }
            payload["optics"] = optics_payload
        if "gas" in request.channels:
            gas_entries = [
                (species_id, shadow_name)
                for species_id in range(self.gas_concentration.shape[0])
                for shadow_name in [self._shadow_gas_name(species_id)]
                if shadow_name
            ]
            payload["gas"] = {
                "origin": [gas_world_x0, gas_world_y0],
                "size": [gas_width, gas_height],
                "grid": "gas",
                "species": {
                    name: gas_species_source(species_id) for species_id, name in gas_entries
                },
            }
        return payload

    def _gas_window_for_cell_rect(self, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        gx0 = min(self.gas_width, max(0, x0 // self.gas_cell_size))
        gy0 = min(self.gas_height, max(0, y0 // self.gas_cell_size))
        gx1 = min(self.gas_width, max(gx0, (x1 + self.gas_cell_size - 1) // self.gas_cell_size))
        gy1 = min(self.gas_height, max(gy0, (y1 + self.gas_cell_size - 1) // self.gas_cell_size))
        return (gx0, gy0, gx1, gy1)

    def _apply_page_stripe(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        if self._gpu_pipeline_available(
            self.page_stripe_pipeline,
            "page stripe",
            require=self.simulation_backend == "gpu",
        ):
            self.page_stripe_pipeline.apply(self, update, payload)
        else:
            self._require_cpu_oracle_backend("page stripe")
            self.page_stripe_pipeline.last_backend = "cpu"
            self.page_stripe_pipeline.last_cpu_mirror_downloaded = True
            self._apply_page_stripe_dense_cpu(update, payload)

        self._normalize_page_stripe_cell_runtime(update)

        runtime_payload = payload.get("runtime")
        self._merge_island_runtime_payload(runtime_payload, update=update, payload=payload)
        if self.page_stripe_pipeline.last_cpu_mirror_downloaded:
            self._queue_loaded_collapse_pending_regions(update)
        else:
            self._queue_loaded_collapse_pending_regions_from_payload(update, payload)
        self._mark_loaded_page_stripe_active(update)
        if self.page_stripe_pipeline.last_cpu_mirror_downloaded:
            self._rebuild_island_records()
        self._apply_page_stripe_entity_placeholder_runtime(
            update,
            None
            if runtime_payload is None
            else runtime_payload.get("entity_placeholder_entity_id"),
        )
        self._invalidate_gpu_authoritative_resources("active_meta", "active_tile_ttl", "active_chunk_mask")

    def _apply_page_stripe_dense_cpu(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        gas_ranges = self._stripe_buffer_ranges(update, gas_grid=True)
        cell_payload = payload["cell"]
        gas_payload = payload["gas"]
        cell_axis = 1 if update.axis == "x" else 0
        cell_dose_axis = 2 if update.axis == "x" else 1

        self._write_stripe_array(self.material_id, update, cell_payload["material_id"], stripe_axis=cell_axis)
        self._write_stripe_array(self.phase, update, cell_payload["phase"], stripe_axis=cell_axis)
        self._write_stripe_array(self.cell_flags, update, cell_payload["cell_flags"], stripe_axis=cell_axis)
        self._write_stripe_array(self.velocity, update, cell_payload["velocity"], stripe_axis=cell_axis)
        self._write_stripe_array(self.cell_temperature, update, cell_payload["cell_temperature"], stripe_axis=cell_axis)
        self._write_stripe_array(self.timer_pack, update, cell_payload["timer_pack"], stripe_axis=cell_axis)
        self._write_stripe_array(self.integrity, update, cell_payload["integrity"], stripe_axis=cell_axis)
        self._write_stripe_array(self.island_id, update, cell_payload["island_id"], stripe_axis=cell_axis)
        self._write_stripe_array(self.entity_id, update, cell_payload["entity_id"], stripe_axis=cell_axis)
        self._write_stripe_array(
            self.placeholder_displaced_material,
            update,
            cell_payload["placeholder_displaced_material"],
            stripe_axis=cell_axis,
        )
        self._write_stripe_array(
            self.collapse_delay_pending,
            update,
            np.asarray(cell_payload["collapse_delay_pending"], dtype=np.bool_),
            stripe_axis=cell_axis,
        )
        self._write_stripe_array(
            self.visible_illumination,
            update,
            cell_payload["visible_illumination"],
            stripe_axis=cell_axis,
        )
        self._write_stripe_array(
            self.cell_optical_dose,
            update,
            cell_payload["cell_optical_dose"],
            stripe_axis=cell_dose_axis,
        )

        self._write_stripe_array(
            self.ambient_temperature,
            update,
            gas_payload["ambient_temperature"],
            stripe_axis=1 if update.axis == "x" else 0,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.flow_velocity,
            update,
            gas_payload["flow_velocity"],
            stripe_axis=1 if update.axis == "x" else 0,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.pressure_ping,
            update,
            gas_payload["pressure_ping"],
            stripe_axis=1 if update.axis == "x" else 0,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.gas_concentration,
            update,
            gas_payload["gas_concentration"],
            stripe_axis=2 if update.axis == "x" else 1,
            ranges=gas_ranges,
        )
        self._write_stripe_array(
            self.gas_optical_dose,
            update,
            gas_payload["gas_optical_dose"],
            stripe_axis=2 if update.axis == "x" else 1,
            ranges=gas_ranges,
        )

    def _queue_loaded_collapse_pending_regions_from_payload(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
    ) -> None:
        if update.kind != "load":
            return
        pending_payload = np.asarray(payload["cell"]["collapse_delay_pending"], dtype=np.bool_)
        offset = 0
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            span = int(end) - int(start)
            if span <= 0:
                continue
            if update.axis == "x":
                pending = pending_payload[:, offset : offset + span]
                ys, xs = np.nonzero(pending)
                if ys.size != 0:
                    self.collapse_deferred_regions.append(
                        (
                            max(0, start + int(xs.min()) - 1),
                            max(0, int(ys.min()) - 1),
                            min(self.width, start + int(xs.max()) + 2),
                            min(self.height, int(ys.max()) + 2),
                        )
                    )
            else:
                pending = pending_payload[offset : offset + span, :]
                ys, xs = np.nonzero(pending)
                if ys.size != 0:
                    self.collapse_deferred_regions.append(
                        (
                            max(0, int(xs.min()) - 1),
                            max(0, start + int(ys.min()) - 1),
                            min(self.width, int(xs.max()) + 2),
                            min(self.height, start + int(ys.max()) + 2),
                        )
                    )
            offset += span

    def _advance_paging(self, center_x: int, center_y: int) -> list[PageStripeUpdate]:
        force_sources = [
            replace(force_source, world_x=self._force_source_world_position(force_source)[0], world_y=self._force_source_world_position(force_source)[1])
            for force_source in self.force_sources
        ]
        updates = self.focus_paging(center_x, center_y)
        if not updates:
            return []
        self.bridge_frame_paging_updates.extend(
            PageStripeUpdate(**asdict(update)) for update in updates
        )
        for update in updates:
            if update.kind == "save":
                self.page_store.save(update, self.capture_page_stripe(update))
                self._clear_saved_page_stripe_runtime_state(update)
        for update in updates:
            if update.kind != "load":
                continue
            payload = self.page_store.load(update)
            if payload is None:
                payload = self._default_page_stripe_payload(update)
            self._apply_page_stripe(update, payload)
            self._record_bridge_page_stripe(update, payload)
        if force_sources:
            self._sync_force_sources(force_sources)
        return updates

    def _prepare_bridge_frame_inputs(self) -> None:
        pending_placeholder_dirty_rects = list(self._pending_placeholder_dirty_rects)
        self._clear_bridge_frame_inputs(keep_commands=False, prepared=True)
        if pending_placeholder_dirty_rects:
            self.bridge_frame_placeholder_dirty_rects.extend(pending_placeholder_dirty_rects)
            self._pending_placeholder_dirty_rects.clear()

    def _needs_pre_simulation_bridge_sync(self, *, frame_input: WorldFrameInput | None) -> bool:
        if self.simulation_backend != "gpu":
            return False
        return bool(
            frame_input is not None
            or self.bridge_frame_placeholders
            or self.bridge_frame_placeholder_dirty_rects
            or self.bridge_frame_paging_updates
            or self.bridge_frame_page_stripes
            or self._gpu_cpu_dirty_resources
        )

    def _sync_pre_simulation_bridge_without_debug_upload(self) -> None:
        try:
            self.bridge.sync_world(self, upload_debug_texture=False)
        except TypeError as exc:
            if "upload_debug_texture" not in str(exc):
                raise
            self.bridge.sync_world(self)

    def _clear_bridge_frame_inputs(self, *, keep_commands: bool, prepared: bool) -> None:
        if not keep_commands:
            self.bridge_frame_commands.clear()
        self.bridge_frame_readback_requests.clear()
        self.bridge_frame_placeholders.clear()
        self.bridge_frame_placeholder_dirty_rects.clear()
        self.bridge_frame_paging_updates.clear()
        self.bridge_frame_page_stripes.clear()
        self._bridge_inputs_prepared = prepared

    def _mark_active_rect_runtime(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        tile_padding: int = 0,
    ) -> None:
        self._mark_active_rects_runtime([(x0, y0, x1, y1, tile_padding)])

    def _mark_active_rects_runtime(
        self,
        rects: list[tuple[int, int, int, int] | tuple[int, int, int, int, int]],
    ) -> None:
        if not rects:
            return
        if self.simulation_backend == "gpu" and self._world_simulation_frame_active:
            if not self.bridge.mark_active_rects(self, rects):
                self._require_gpu_stage("active scheduler region marking")
            return
        for rect in rects:
            if len(rect) == 4:
                x0, y0, x1, y1 = rect
                tile_padding = 0
            else:
                x0, y0, x1, y1, tile_padding = rect
            self.active.mark_rect(int(x0), int(y0), int(x1), int(y1), tile_padding=int(tile_padding))
        if self.simulation_backend == "gpu":
            self._invalidate_gpu_authoritative_resources("active_meta", "active_tile_ttl", "active_chunk_mask")

    def _sync_entity_placeholders(self, placeholders: list[EntityPlaceholder]) -> None:
        self.bridge_frame_placeholders.extend(replace(placeholder) for placeholder in placeholders)
        current_cells = {
            cell: entity_id
            for entity_id, cells in self.entity_placeholders.items()
            for cell in cells
        }
        next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
        for placeholder in placeholders:
            for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
                for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                    if not self.in_bounds(x, y):
                        continue
                    next_cells[(x, y)] = placeholder

        changed_cells: set[tuple[int, int]] = set()
        for cell, entity_id in current_cells.items():
            next_placeholder = next_cells.get(cell)
            if next_placeholder is None or next_placeholder.entity_id != entity_id:
                changed_cells.add(cell)
        for cell, placeholder in next_cells.items():
            if current_cells.get(cell) != placeholder.entity_id:
                changed_cells.add(cell)

        if self._gpu_pipeline_available(
            self.placeholder_pipeline,
            "placeholder",
            require=self.simulation_backend == "gpu",
        ):
            if current_cells or next_cells:
                if self.simulation_backend == "gpu" and self._world_simulation_frame_active and (
                    not self._bridge_inputs_prepared or self._gpu_cpu_dirty_resources
                ):
                    self._sync_pre_simulation_bridge_without_debug_upload()
                    self._gpu_cpu_dirty_resources.clear()
                    self._bridge_inputs_prepared = True
                self.placeholder_pipeline.apply(self, placeholders)
                if self.placeholder_pipeline.last_cpu_mirror_downloaded:
                    self._rebuild_entity_placeholder_index()
                else:
                    next_entity_cells: dict[int, set[tuple[int, int]]] = {}
                    for cell, placeholder in next_cells.items():
                        next_entity_cells.setdefault(int(placeholder.entity_id), set()).add(cell)
                    self.entity_placeholders = next_entity_cells
            else:
                self.entity_placeholders.clear()
                self.placeholder_pipeline.last_backend = "idle"
            for x, y in sorted(changed_cells):
                self._mark_active_rect_runtime(
                    max(0, x - 1),
                    max(0, y - 1),
                    min(self.width, x + 2),
                    min(self.height, y + 2),
                )
            self.bridge_frame_placeholder_dirty_rects.extend((x, y, x + 1, y + 1) for x, y in sorted(changed_cells))
            return

        self._require_cpu_oracle_backend("placeholder")
        self.placeholder_pipeline.last_backend = "cpu" if (current_cells or next_cells) else "idle"
        for cell, entity_id in current_cells.items():
            next_placeholder = next_cells.get(cell)
            x, y = cell
            material_id = int(self.material_id[y, x])
            if (
                next_placeholder is not None
                and next_placeholder.entity_id == entity_id
                and material_id > 0
                and self._shadow_material_is_placeholder(material_id)
            ):
                continue
            self._release_entity_placeholder_cell(x, y, entity_id)

        next_entity_cells: dict[int, set[tuple[int, int]]] = {}
        for cell, placeholder in next_cells.items():
            x, y = cell
            material_id = int(self.material_id[y, x])
            if (
                current_cells.get(cell) == placeholder.entity_id
                and material_id > 0
                and self._shadow_material_is_placeholder(material_id)
            ):
                next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
                self.entity_id[y, x] = placeholder.entity_id
                continue
            if self._occupy_entity_placeholder_cell(x, y, placeholder):
                next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
        self.entity_placeholders = next_entity_cells
        self.bridge_frame_placeholder_dirty_rects.extend((x, y, x + 1, y + 1) for x, y in sorted(changed_cells))

    def _sync_force_sources(self, force_sources: list[ForceSource]) -> None:
        self.force_sources = [
            self._normalize_runtime_force_source(
                force_source
                if isinstance(force_source, ForceSource)
                else ForceSource(**force_source)
            )
            for force_source in force_sources
        ]
        for force_source in self.force_sources:
            radius = int(np.ceil(force_source.radius))
            x = int(round(force_source.x))
            y = int(round(force_source.y))
            self._mark_active_rect_runtime(
                max(0, x - radius),
                max(0, y - radius),
                min(self.width, x + radius + 1),
                min(self.height, y + radius + 1),
            )

    def _append_force_source_immediate(self, force_source: ForceSource) -> None:
        self.force_sources.append(self._normalize_runtime_force_source(force_source))
        radius = int(np.ceil(force_source.radius))
        x = int(round(self.force_sources[-1].x))
        y = int(round(self.force_sources[-1].y))
        self._mark_active_rect_runtime(
            max(0, x - radius),
            max(0, y - radius),
            min(self.width, x + radius + 1),
            min(self.height, y + radius + 1),
        )

    def _sync_persistent_emitters(self, emitters: list[dict[str, object]]) -> None:
        self.persistent_emitters = [dict(emitter) for emitter in emitters]
        for emitter in self.persistent_emitters:
            radius = int(max(0, round(float(emitter["range_cells"]))))
            x = int(emitter["origin"][0])
            y = int(emitter["origin"][1])
            self._mark_active_rect_runtime(
                max(0, x - radius),
                max(0, y - radius),
                min(self.width, x + radius + 1),
                min(self.height, y + radius + 1),
            )

    def _append_transient_light_emitter_immediate(self, emitter: dict[str, object]) -> None:
        record = dict(emitter)
        self.emitters.append(record)
        radius = int(max(0, round(float(record["range_cells"]))))
        x = int(record["origin"][0])
        y = int(record["origin"][1])
        self._mark_active_rect_runtime(
            max(0, x - radius),
            max(0, y - radius),
            min(self.width, x + radius + 1),
            min(self.height, y + radius + 1),
        )

    def _record_bridge_page_stripe(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        self.bridge_frame_page_stripes.append((PageStripeUpdate(**asdict(update)), deepcopy(payload)))

    def _release_entity_placeholder_cell(self, x: int, y: int, entity_id: int) -> None:
        if not self.in_bounds(x, y):
            return
        if int(self.entity_id[y, x]) != entity_id:
            return
        self.entity_id[y, x] = 0
        material_id = int(self.material_id[y, x])
        if material_id <= 0 or not self._shadow_material_is_placeholder(material_id):
            return
        displaced_material = int(self.placeholder_displaced_material[y, x])
        if displaced_material > 0:
            self.material_id[y, x] = displaced_material
            self.phase[y, x] = int(Phase.LIQUID)
            self.cell_flags[y, x] = 0
            self.timer_pack[y, x] = 0
            shadow_integrity = self._shadow_material_base_integrity(displaced_material)
            self.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
            self.placeholder_displaced_material[y, x] = 0
            self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
            return
        self.clear_cell(x, y, mark_dirty=False)

    def _mirror_release_entity_placeholder_cell(self, x: int, y: int, entity_id: int) -> None:
        if not self.in_bounds(x, y):
            return
        if int(self.entity_id[y, x]) != entity_id:
            return
        self._invalidate_gpu_authoritative_cell_resources()
        self.entity_id[y, x] = 0
        material_id = int(self.material_id[y, x])
        if material_id <= 0 or not self._shadow_material_is_placeholder(material_id):
            return
        displaced_material = int(self.placeholder_displaced_material[y, x])
        if displaced_material > 0:
            self.material_id[y, x] = displaced_material
            self.phase[y, x] = int(Phase.LIQUID)
            self.cell_flags[y, x] = 0
            self.timer_pack[y, x] = 0
            shadow_integrity = self._shadow_material_base_integrity(displaced_material)
            self.integrity[y, x] = float(shadow_integrity) if shadow_integrity is not None else 0.0
            self.placeholder_displaced_material[y, x] = 0
            self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
            return
        self.clear_cell(x, y, mark_dirty=False)

    def _resolve_sanctioned_material_id(self, name: str) -> int:
        self.bridge.sync_rule_tables(self)
        material_table = self.bridge.shadow_typed_tables.get("material_table")
        if material_table is None:
            return 0
        canonical_names = [str(name)]
        alias = BASE_MATERIAL_RUNTIME_ALIASES.get(str(name))
        if alias is not None and alias != name:
            canonical_names.append(alias)
        for candidate_name in canonical_names:
            material_id = int(typed_material_id(material_table, candidate_name))
            if material_id <= 0 or material_id >= int(material_table.shape[0]):
                continue
            if int(material_table[material_id]["name_hash"]) == 0:
                continue
            return material_id
        return 0

    def _shadow_material_id_by_name(self, name: str | None) -> int:
        canonical_name = self._canonical_material_input_name(name)
        if canonical_name is None:
            return 0
        materials_payload = self._shadow_material_payload()
        if materials_payload is not None:
            for item in materials_payload:
                if self._canonical_material_input_name(item.get("name")) != canonical_name:
                    continue
                return int(item.get("material_id", 0))
        material_table = self.bridge.shadow_typed_tables.get("material_table")
        if material_table is None:
            return 0
        return int(typed_material_id(material_table, canonical_name))

    def _resolve_sanctioned_placeholder_material_id(self, name: str) -> int:
        material_id = self._resolve_sanctioned_material_id(name)
        if not self._shadow_material_is_placeholder(material_id):
            return 0
        return material_id

    def _resolve_sanctioned_light_id(self, name: str) -> int:
        self.bridge.sync_rule_tables(self)
        light_table = self.bridge.shadow_typed_tables.get("light_table")
        if light_table is None:
            return -1
        light_id = int(typed_light_id(light_table, name))
        if light_id < 0 or light_id >= int(light_table.shape[0]):
            return -1
        if int(light_table[light_id]["name_hash"]) == 0:
            return -1
        return light_id

    def _resolve_sanctioned_gas_id(self, name: str) -> int:
        self.bridge.sync_rule_tables(self)
        gas_table = self.bridge.shadow_typed_tables.get("gas_table")
        if gas_table is None:
            return -1
        species_id = int(typed_gas_id(gas_table, name))
        if species_id < 0 or species_id >= int(gas_table.shape[0]):
            return -1
        if int(gas_table[species_id]["name_hash"]) == 0:
            return -1
        return species_id

    def _shadow_material_row_valid(self, material_id: int) -> bool:
        material_table = self.bridge.shadow_typed_tables.get("material_table")
        if material_table is None:
            return True
        if material_id <= 0 or material_id >= int(material_table.shape[0]):
            return False
        return int(material_table[int(material_id)]["name_hash"]) != 0

    def _shadow_gas_row_valid(self, species_id: int) -> bool:
        gas_table = self.bridge.shadow_typed_tables.get("gas_table")
        if gas_table is None:
            return True
        if species_id < 0 or species_id >= int(gas_table.shape[0]):
            return False
        return int(gas_table[int(species_id)]["name_hash"]) != 0

    def _shadow_light_row_valid(self, light_id: int) -> bool:
        light_table = self.bridge.shadow_typed_tables.get("light_table")
        if light_table is None:
            return True
        if light_id < 0 or light_id >= int(light_table.shape[0]):
            return False
        return int(light_table[int(light_id)]["name_hash"]) != 0

    def _shadow_material_def(self, material_id: int) -> MaterialDef | None:
        if not self._shadow_material_row_valid(int(material_id)):
            return None
        for item in self._shadow_material_payload():
            if int(item.get("material_id", 0)) == int(material_id):
                return self._coerce_material_def(item)
        return None

    def _shadow_light_type_def(self, light_id: int) -> LightTypeDef | None:
        if not self._shadow_light_row_valid(int(light_id)):
            return None
        for item in self._shadow_light_type_payload():
            if int(item.get("light_type_id", -1)) == int(light_id):
                return self._coerce_light_type_def(item)
        return None

    def _shadow_gas_species_def(self, species_id: int) -> GasSpeciesDef | None:
        if not self._shadow_gas_row_valid(int(species_id)):
            return None
        for item in self._shadow_gas_species_payload():
            if int(item.get("species_id", -1)) == int(species_id):
                return self._coerce_gas_species_def(item)
        return None

    def _shadow_material_optics_def(self, material_name: str, light_type: str) -> MaterialOpticsDef | None:
        payload = self.bridge.shadow_tables.get("optics")
        if payload is not None:
            for item in payload:
                if str(item.get("material_name", "")) == material_name and str(item.get("light_type", "")) == light_type:
                    return self._coerce_material_optics_def(item)
            return None
        optics_table = self.bridge.shadow_typed_tables.get("optics_table")
        material_id = self._resolve_sanctioned_material_id(material_name)
        light_id = self._resolve_sanctioned_light_id(light_type)
        if material_id <=0 or light_id <0:
            return None
        if optics_table is not None:
            for row in optics_table:
                if int(row["material_id"]) == material_id and int(row["light_type_id"]) == light_id:
                    return MaterialOpticsDef(
                    material_name=material_name,
                    light_type=light_type,
                    absorption=float(row["absorption"]),
                    scattering=float(row["scattering"]),
                    refraction=float(row["refraction"]),
                    )
        return None
    def _shadow_material_name(self, material_id: int) -> str | None:
        material = self._shadow_material_def(int(material_id))
        if material is not None and material.name:
            return str(material.name)
        if not self._shadow_material_row_valid(int(material_id)):
            return None
        if self._shadow_has_table_payload("materials"):
            return None
        if 0 <= int(material_id) < len(self.material_name_by_id) and self.material_name_by_id[int(material_id)]:
            return self.material_name_by_id[int(material_id)]
        return None

    def _shadow_gas_name(self, species_id: int) -> str | None:
        gas = self._shadow_gas_species_def(int(species_id))
        if gas is not None and gas.name:
            return str(gas.name)
        if not self._shadow_gas_row_valid(int(species_id)):
            return None
        if self._shadow_has_table_payload("gases"):
            return None

        return self.gas_name_by_id[int(species_id)]
        return None

    def _shadow_light_name(self, light_id: int) -> str | None:
        light = self._shadow_light_type_def(int(light_id))
        if light is not None and light.name:
            return str(light.name)
        if not self._shadow_light_row_valid(int(light_id)):
            return None
        if self._shadow_has_table_payload("lights"):
            return None
        if 0 <= int(light_id) < len(self.light_name_by_id) and self.light_name_by_id[int(light_id)]:
            return self.light_name_by_id[int(light_id)]
        return None

    def _shadow_light_default_range(self, light_id: int) -> int | None:
        light_table = self.bridge.shadow_typed_tables.get("light_table")
        if light_table is not None and 0 <= int(light_id) < int(light_table.shape[0]):
            row = light_table[int(light_id)]
            if int(row["name_hash"]) == 0:
                return None
            return int(row["default_range"])
        light = self._shadow_light_type_def(int(light_id))
        if light is not None:
            return int(light.default_range)
        if self._shadow_has_table_payload("lights"):
            return None
        if 0 <= int(light_id) < self.light_default_range.shape[0]:
            return int(self.light_default_range[int(light_id)])
        return None

    def _shadow_light_dose_channel(self, light_id: int) -> int | None:
        light_table = self.bridge.shadow_typed_tables.get("light_table")
        if light_table is not None and 0 <= int(light_id) < int(light_table.shape[0]):
            row = light_table[int(light_id)]
            if int(row["name_hash"]) == 0:
                return None
            return int(row["dose_channel_id"])
        light = self._shadow_light_type_def(int(light_id))
        if light is not None:
            return int(light.dose_channel_id)
        if self._shadow_has_table_payload("lights"):
            return None
        if 0 <= int(light_id) < self.light_dose_channel.shape[0]:
            return int(self.light_dose_channel[int(light_id)])
        return None

    def _shadow_light_color(self, light_id: int) -> np.ndarray | None:
        light_table = self.bridge.shadow_typed_tables.get("light_table")
        if light_table is not None and 0 <= int(light_id) < int(light_table.shape[0]):
            row = light_table[int(light_id)]
            if int(row["name_hash"]) == 0:
                return None
            return np.asarray(row["color"], dtype=np.float32)
        light = self._shadow_light_type_def(int(light_id))
        if light is not None:
            return np.asarray(light.color, dtype=np.float32)
        if self._shadow_has_table_payload("lights"):
            return None
        if 0 <= int(light_id) < self.light_color.shape[0]:
            return np.asarray(self.light_color[int(light_id)], dtype=np.float32)
        return None

    def _shadow_light_name_and_range(self, light_id: int) -> tuple[str, int] | None:
        light_name = self._shadow_light_name(int(light_id))
        if light_name is None:
            return None
        default_range = self._shadow_light_default_range(int(light_id))
        if default_range is None:
            return None
        return (light_name, default_range)

    def _shadow_material_default_phase(self, material_id: int) -> int | None:
        material_table = self.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= int(material_id) < int(material_table.shape[0]):
            row = material_table[int(material_id)]
            if int(row["name_hash"]) == 0:
                return None
            return int(row["default_phase"])
        shadow_material = self._shadow_material_def(int(material_id))
        if shadow_material is not None:
            return int(shadow_material.default_phase)
        if self._shadow_has_table_payload("materials"):
            return None
        if 0 <= int(material_id) < self.material_default_phase.shape[0]:
            return int(self.material_default_phase[int(material_id)])
        return None

    def _shadow_material_base_integrity(self, material_id: int) -> float | None:
        material_table = self.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= int(material_id) < int(material_table.shape[0]):
            row = material_table[int(material_id)]
            if int(row["name_hash"]) == 0:
                return None
            return float(row["base_integrity"])
        shadow_material = self._shadow_material_def(int(material_id))
        if shadow_material is not None:
            return float(shadow_material.base_integrity)
        if self._shadow_has_table_payload("materials"):
            return None
        if 0 <= int(material_id) < self.material_base_integrity.shape[0]:
            return float(self.material_base_integrity[int(material_id)])
        return None

    def _shadow_material_spawn_temperature(self, material_id: int) -> float | None:
        material_table = self.bridge.shadow_typed_tables.get("material_table")
        if material_table is not None and 0 <= int(material_id) < int(material_table.shape[0]):
            row = material_table[int(material_id)]
            if int(row["name_hash"]) == 0:
                return None
            value = float(row["spawn_temperature"])
            return None if np.isnan(value) else value
        shadow_material = self._shadow_material_def(int(material_id))
        if shadow_material is not None and shadow_material.spawn_temperature is not None:
            return float(shadow_material.spawn_temperature)
        if self._shadow_has_table_payload("materials"):
            return None
        if 0 <= int(material_id) < self.material_spawn_temperature.shape[0]:
            value = float(self.material_spawn_temperature[int(material_id)])
            return None if np.isnan(value) else value
        return None

    def _shadow_condense_target_material_id(self, species_id: int) -> int:
        gas_table = self.bridge.shadow_typed_tables.get("gas_table")
        if gas_table is not None and 0 <= int(species_id) < int(gas_table.shape[0]):
            row = gas_table[int(species_id)]
            if int(row["name_hash"]) != 0:
                return int(row["condense_to_material_id"])
        gas = self._shadow_gas_species_def(int(species_id))
        if gas is None or gas.condense_to_material is None:
            return 0
        return self._shadow_material_id_by_name(gas.condense_to_material)

    def _shadow_material_is_placeholder(self, material_id: int) -> bool:
        shadow_material = self._shadow_material_def(int(material_id))
        if shadow_material is not None:
            return shadow_material.render_group == "placeholder" or "placeholder" in shadow_material.tags
        if self.bridge.shadow_typed_tables.get("material_table") is not None:
            return False
        if self._shadow_has_table_payload("materials"):
            return False
        if 0 <= int(material_id) < self.material_is_placeholder.shape[0]:
            return bool(self.material_is_placeholder[int(material_id)])
        return False

    def _material_placeholder_mask(self, material_id: np.ndarray) -> np.ndarray:
        ids = np.asarray(material_id, dtype=np.int64)
        mask = np.zeros(ids.shape, dtype=np.bool_)
        valid = (ids >= 0) & (ids < int(self.material_is_placeholder.shape[0]))
        if np.any(valid):
            mask[valid] = self.material_is_placeholder[ids[valid]]
        return mask

    def _shadow_material_is_plant(self, material_id: int) -> bool:
        shadow_material = self._shadow_material_def(int(material_id))
        if shadow_material is not None:
            return shadow_material.render_group == "plant" or "plant" in shadow_material.tags
        if self.bridge.shadow_typed_tables.get("material_table") is not None:
            return False
        if self._shadow_has_table_payload("materials"):
            return False
        if 0 <= int(material_id) < self.material_is_plant.shape[0]:
            return bool(self.material_is_plant[int(material_id)])
        return False

    def _shadow_reaction_action(self, index: int) -> ReactionAction | None:
        if index ==0:
            return self.rulebook.reaction_actions[0] if self.rulebook.reaction_actions else ReactionAction(ReactionType.NONE)
        payload = self._shadow_reaction_payload()
        actions = payload.get("actions", [])
        if index >0 and index <= len(actions):
            return self._coerce_reaction_action(actions[index -1])
        if self._shadow_has_table_payload("reactions"):
            return None
        if index >=0 and index < len(self.rulebook.reaction_actions):
            return self.rulebook.reaction_actions[index]
        return None

    def _reaction_rule_list(self, rule_set: str) -> list[PairReactionRule] | list[SelfReactionRule]:
        normalized = str(rule_set)
        payload = self._shadow_reaction_payload()
        if payload is not None:
            rules_payload = payload.get("rules", {})
            entries = list(rules_payload.get(normalized, []))
            if normalized == "self_rules":
                return [self._coerce_self_reaction_rule(entry) for entry in entries]
            if normalized in PAIR_REACTION_RULE_SET_NAMES:
                return [self._coerce_pair_reaction_rule(entry) for entry in entries]
        if self._shadow_has_table_payload("reactions"):
            return []
        if normalized == "material_material":
            return self.rulebook.material_material_rules
        if normalized == "material_gas":
            return self.rulebook.material_gas_rules
        if normalized == "material_light":
            return self.rulebook.material_light_rules
        if normalized == "gas_gas":
            return self.rulebook.gas_gas_rules
        if normalized == "gas_light":
            return self.rulebook.gas_light_rules
        if normalized == "self_rules":
            return self.rulebook.self_rules
        raise KeyError(rule_set)

    def _set_reaction_rule_list(self, rule_set: str, entries: list[dict[str, Any]] | list[PairReactionRule] | list[SelfReactionRule]) -> None:
        normalized = str(rule_set)
        if normalized == "self_rules":
            normalized_entries = [self._coerce_self_reaction_rule(entry) for entry in entries]
            self.rulebook.self_rules = normalized_entries
            return
        normalized_entries = [self._coerce_pair_reaction_rule(entry) for entry in entries]
        if normalized == "material_material":
            self.rulebook.material_material_rules = normalized_entries
            return
        if normalized == "material_gas":
            self.rulebook.material_gas_rules = normalized_entries
            return
        if normalized == "material_light":
            self.rulebook.material_light_rules = normalized_entries
            return
        if normalized == "gas_gas":
            self.rulebook.gas_gas_rules = normalized_entries
            return
        if normalized == "gas_light":
            self.rulebook.gas_light_rules = normalized_entries
            return
        raise KeyError(rule_set)

    def _set_reaction_rules_payload(self, rules_payload: dict[str, list[dict[str, Any]]]) -> None:
        for rule_set in REACTION_RULE_SET_NAMES:
            self._set_reaction_rule_list(str(rule_set), list(rules_payload.get(str(rule_set), [])))

    @staticmethod
    def _remap_reaction_payload_result_actions(
        rules_payload: dict[str, list[dict[str, Any]]],
        *,
        deleted_action_index: int,
    ) -> None:
        for rule_set in PAIR_REACTION_RULE_SET_NAMES:
            for rule in rules_payload.get(str(rule_set), []):
                result_action = int(rule.get("result_action", -1))
                if result_action == deleted_action_index:
                    rule["result_action"] = 0
                elif result_action > deleted_action_index:
                    rule["result_action"] = result_action - 1

    @staticmethod
    def _remap_material_payload_reaction_slots(
        materials_payload: list[dict[str, Any]],
        *,
        deleted_action_index: int,
    ) -> None:
        for material in materials_payload:
            slots = list(material.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
            remapped_slots: list[int] = []
            for slot in slots[:8]:
                action_index = int(slot)
                if action_index == deleted_action_index:
                    remapped_slots.append(-1)
                elif action_index > deleted_action_index:
                    remapped_slots.append(action_index - 1)
                else:
                    remapped_slots.append(action_index)
            if len(remapped_slots) < 8:
                remapped_slots.extend([-1] * (8 - len(remapped_slots)))
            material["reaction_slots"] = tuple(remapped_slots)

    @staticmethod
    def _clamp_material_payload_reaction_slots(
        materials_payload: list[dict[str, Any]],
        *,
        action_count: int,
    ) -> None:
        for material in materials_payload:
            slots = list(material.get("reaction_slots", (-1, -1, -1, -1, -1, -1, -1, -1)))
            clamped_slots: list[int] = []
            for slot in slots[:8]:
                action_index = int(slot)
                if action_index < -1 or action_index >= action_count:
                    clamped_slots.append(-1)
                else:
                    clamped_slots.append(action_index)
            if len(clamped_slots) < 8:
                clamped_slots.extend([-1] * (8 - len(clamped_slots)))
            material["reaction_slots"] = tuple(clamped_slots)

    def _shadow_reaction_rule(self, rule_set: str, index: int) -> PairReactionRule | SelfReactionRule | None:
        payload = self._shadow_reaction_payload()
        normalized = str(rule_set)
        rules = payload.get("rules", {})
        entries = rules.get(normalized, [])
        if (0 <= index) and index < len(entries):
            if normalized == "self_rules":
                return self._coerce_self_reaction_rule(entries[index])
            return self._coerce_pair_reaction_rule(entries[index])
        if self._shadow_has_table_payload("reactions"):
            return None
        if (0 <= index) and index < len(rules_list):
            return rules_list[index]
        return None

    def _occupy_entity_placeholder_cell(self, x: int, y: int, placeholder: EntityPlaceholder) -> bool:
        if not self.in_bounds(x, y):
            return False
        placeholder_material_id = self._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
        if placeholder_material_id <= 0:
            return False
        material_id = int(self.material_id[y, x])
        if material_id != 0 and int(self.phase[y, x]) != int(Phase.LIQUID):
            return False
        self.set_cell_by_id(x, y, placeholder_material_id, mark_dirty=False)
        self.entity_id[y, x] = placeholder.entity_id
        self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
        return True

    def _mirror_occupy_entity_placeholder_cell(self, x: int, y: int, placeholder: EntityPlaceholder) -> bool:
        if not self.in_bounds(x, y):
            return False
        placeholder_material_id = self._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
        if placeholder_material_id <= 0:
            return False
        material_id = int(self.material_id[y, x])
        if material_id != 0 and int(self.phase[y, x]) != int(Phase.LIQUID):
            return False
        self.set_cell_by_id(x, y, placeholder_material_id, mark_dirty=False)
        self.entity_id[y, x] = placeholder.entity_id
        self._mark_active_rect_runtime(max(0, x - 1), max(0, y - 1), min(self.width, x + 2), min(self.height, y + 2))
        return True

    def _frame_entities_to_placeholders_and_observations(
        self,
        entities: list[EntityState],
    ) -> tuple[list[EntityPlaceholder], list[ObservationTarget]]:
        placeholders = [
            EntityPlaceholder(
                entity_id=entity.entity_id,
                x=entity.x,
                y=entity.y,
                width=entity.width,
                height=entity.height,
                material=entity.placeholder_material,
                world_x=entity.world_x,
                world_y=entity.world_y,
            )
            for entity in entities
        ]
        observation_targets = [
            ObservationTarget(
                observer_id=entity.entity_id,
                entity_id=entity.entity_id,
                channels=entity.observe_channels,
                pad_cells=entity.observe_pad_cells,
                width=entity.observe_width,
                height=entity.observe_height,
                label=entity.observe_label,
            )
            for entity in entities
            if entity.observe_channels
        ]
        return placeholders, observation_targets

    def _runtime_entities_to_immediate_observation_targets(
        self,
        entities: list[EntityState],
    ) -> list[ObservationTarget]:
        targets: list[ObservationTarget] = []
        for entity in entities:
            if not entity.observe_channels:
                continue
            if entity.world_x is not None and entity.world_y is not None:
                world_x = int(entity.world_x)
                world_y = int(entity.world_y)
            else:
                world_x, world_y = self._buffer_to_world_position((int(entity.x), int(entity.y)))
            entity_width = max(1, int(entity.width))
            entity_height = max(1, int(entity.height))
            center_x = int((world_x + world_x + entity_width - 1) // 2)
            center_y = int((world_y + world_y + entity_height - 1) // 2)
            width = int(entity.observe_width) if entity.observe_width is not None else entity_width + int(entity.observe_pad_cells) * 2
            height = int(entity.observe_height) if entity.observe_height is not None else entity_height + int(entity.observe_pad_cells) * 2
            targets.append(
                ObservationTarget(
                    observer_id=int(entity.entity_id),
                    center_x=int(center_x),
                    center_y=int(center_y),
                    width=max(1, int(width)),
                    height=max(1, int(height)),
                    channels=entity.observe_channels,
                    pad_cells=int(entity.observe_pad_cells),
                    label=entity.observe_label,
                )
            )
        return targets

    def _sync_entity_states(self, entities: list[EntityState]) -> tuple[list[EntityPlaceholder], list[ObservationTarget]]:
        self.entity_states = {entity.entity_id: entity for entity in entities}
        placeholders, observation_targets = self._frame_entities_to_placeholders_and_observations(entities)
        return placeholders, observation_targets

    def _sync_entity_observation_specs(self, observations: list[EntityObservationSpec]) -> None:
        observation_by_entity_id = {observation.entity_id: observation for observation in observations}
        self.entity_states = {
            entity_id: replace(
                entity,
                observe_channels=observation.observe_channels if observation is not None else (),
                observe_pad_cells=int(observation.observe_pad_cells) if observation is not None else 0,
                observe_width=None if observation is None else observation.observe_width,
                observe_height=None if observation is None else observation.observe_height,
                observe_label=None if observation is None else observation.observe_label,
            )
            for entity_id, entity in self.entity_states.items()
            for observation in [observation_by_entity_id.get(entity_id)]
        }

    def _patch_entity_states(self, patches: list[EntityStatePatch]) -> None:
        next_entity_states = dict(self.entity_states)
        for patch in patches:
            entity = next_entity_states.get(patch.entity_id)
            if entity is None:
                raise KeyError(patch.entity_id)
            patch_fields = {name: value for name, value in patch.fields.items() if not name.startswith("_")}
            world_x = patch.fields.get(
                "_world_x",
                patch_fields.get("x", entity.world_x if entity.world_x is not None else entity.x),
            )
            world_y = patch.fields.get(
                "_world_y",
                patch_fields.get("y", entity.world_y if entity.world_y is not None else entity.y),
            )
            next_entity_states[patch.entity_id] = self._coerce_entity_state(
                replace(entity, **dict(patch_fields), world_x=int(world_x), world_y=int(world_y))
            )
        self.entity_states = next_entity_states
        placeholders, _ = self._frame_entities_to_placeholders_and_observations(list(self.entity_states.values()))
        self._sync_entity_placeholders(placeholders)

    def _build_preview_entity_placeholders(
        self,
        placeholders: list[EntityPlaceholder],
    ) -> tuple[dict[int, set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
        current_cells = {
            cell: entity_id
            for entity_id, cells in self.entity_placeholders.items()
            for cell in cells
        }
        next_cells: dict[tuple[int, int], EntityPlaceholder] = {}
        for placeholder in placeholders:
            for y in range(placeholder.y, placeholder.y + max(0, placeholder.height)):
                for x in range(placeholder.x, placeholder.x + max(0, placeholder.width)):
                    if not self.in_bounds(x, y):
                        continue
                    next_cells[(x, y)] = placeholder

        released_cells = {
            cell
            for cell, entity_id in current_cells.items()
            if next_cells.get(cell) is None or next_cells[cell].entity_id != entity_id
        }
        next_entity_cells: dict[int, set[tuple[int, int]]] = {}
        for cell, placeholder in next_cells.items():
            if self._preview_can_occupy_placeholder_cell(cell[0], cell[1], placeholder, current_cells, released_cells):
                next_entity_cells.setdefault(placeholder.entity_id, set()).add(cell)
        blocked_cells = {
            cell
            for cells in next_entity_cells.values()
            for cell in cells
        }
        return next_entity_cells, blocked_cells, released_cells

    def _preview_can_occupy_placeholder_cell(
        self,
        x: int,
        y: int,
        placeholder: EntityPlaceholder,
        current_cells: dict[tuple[int, int], int],
        released_cells: set[tuple[int, int]],
    ) -> bool:
        if not self.in_bounds(x, y):
            return False
        placeholder_material_id = self._resolve_sanctioned_placeholder_material_id(str(placeholder.material))
        if placeholder_material_id <= 0:
            return False
        material_id, phase = self._material_state_for_position(x, y, released_cells=released_cells)
        if current_cells.get((x, y)) == placeholder.entity_id and material_id > 0 and self._shadow_material_is_placeholder(material_id):
            return True
        return material_id == 0 or phase == int(Phase.LIQUID)

    def _material_state_for_position(
        self,
        x: int,
        y: int,
        *,
        blocked_cells: set[tuple[int, int]] | None = None,
        released_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        material_id = int(self.material_id[y, x])
        phase = int(self.phase[y, x])
        cell = (x, y)
        if released_cells is not None and cell in released_cells and material_id > 0 and self._shadow_material_is_placeholder(material_id):
            displaced_material = int(self.placeholder_displaced_material[y, x])
            if displaced_material > 0:
                return displaced_material, int(Phase.LIQUID)
            return 0, 0
        if blocked_cells is not None and cell in blocked_cells:
            return int(self.placeholder_material_id), int(Phase.STATIC_SOLID)
        return material_id, phase

    def _build_observation_requests(
        self,
        targets: list[ObservationTarget],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[ReadbackRequest]:
        return [request for _, request in self._build_observation_request_pairs(targets, resolved_targets)]

    def _build_observation_request(
        self,
        target: ObservationTarget,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ReadbackRequest | None:
        center_x = target.center_x
        center_y = target.center_y
        width = target.width
        height = target.height
        if target.target_query_id is not None and (center_x is None or center_y is None or width is None or height is None):
            resolved_target = resolved_targets.get(target.target_query_id)
            if resolved_target is None or resolved_target.status != "resolved" or resolved_target.resolved_world_position is None:
                return None
            if center_x is None:
                center_x = int(resolved_target.resolved_world_position[0]) + int(target.target_dx)
            if center_y is None:
                center_y = int(resolved_target.resolved_world_position[1]) + int(target.target_dy)
            if width is None:
                width = 1 + int(target.pad_cells) * 2
            if height is None:
                height = 1 + int(target.pad_cells) * 2
        if target.entity_id is not None and (center_x is None or center_y is None or width is None or height is None):
            bbox = self._entity_placeholder_bbox(target.entity_id)
            if bbox is None:
                return None
            x0, y0, x1, y1 = self._buffer_bbox_to_world_bbox(bbox)
            if center_x is None:
                center_x = (x0 + x1 - 1) // 2
            if center_y is None:
                center_y = (y0 + y1 - 1) // 2
            if width is None:
                width = (x1 - x0) + target.pad_cells * 2
            if height is None:
                height = (y1 - y0) + target.pad_cells * 2
        if center_x is None or center_y is None:
            return None
        return self._normalize_readback_request(
            ReadbackRequest(
                center_x=center_x,
                center_y=center_y,
                width=max(1, width if width is not None else 1),
                height=max(1, height if height is not None else 1),
                channels=target.channels,
                observer_id=target.observer_id,
                label=target.label,
                target_query_id=target.target_query_id,
                target_dx=int(target.target_dx),
                target_dy=int(target.target_dy),
            )
        )

    def _resolve_readback_requests(
        self,
        requests: list[ReadbackRequest],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[ReadbackRequest]:
        resolved: list[ReadbackRequest] = []
        for request in requests:
            concrete = self._resolve_readback_request(request, resolved_targets)
            if concrete is not None:
                resolved.append(concrete)
        return resolved

    def _resolve_readback_request(
        self,
        request: ReadbackRequest,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ReadbackRequest | None:
        center_x = request.center_x
        center_y = request.center_y
        if request.target_query_id is not None and (center_x is None or center_y is None):
            resolved_target = resolved_targets.get(request.target_query_id)
            if resolved_target is None or resolved_target.status != "resolved" or resolved_target.resolved_world_position is None:
                return None
            if center_x is None:
                center_x = int(resolved_target.resolved_world_position[0]) + int(request.target_dx)
            if center_y is None:
                center_y = int(resolved_target.resolved_world_position[1]) + int(request.target_dy)
        if center_x is None or center_y is None:
            return None
        return self._normalize_readback_request(
            ReadbackRequest(
                request_id=request.request_id,
                center_x=int(center_x),
                center_y=int(center_y),
                width=max(1, int(request.width)),
                height=max(1, int(request.height)),
                channels=request.channels,
                observer_id=request.observer_id,
                label=request.label,
                target_query_id=request.target_query_id,
                target_dx=int(request.target_dx),
                target_dy=int(request.target_dy),
            )
        )

    def _resolve_target_queries(self, queries: list[TargetQuery]) -> dict[str, ResolvedTarget]:
        resolved: dict[str, ResolvedTarget] = {}
        for query in queries:
            resolved[query.query_id] = self._resolve_target_query(query)
        return resolved

    def _resolve_change_intents(
        self,
        intents: list[ChangeIntent],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[dict[str, ResolvedChangeIntent], list[WorldCommand]]:
        resolved: dict[str, ResolvedChangeIntent] = {}
        commands: list[WorldCommand] = []
        for intent in intents:
            resolved_intent = self._resolve_change_intent(intent, resolved_targets)
            resolved[intent.intent_id] = resolved_intent
            commands.extend(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)) for command in resolved_intent.generated_commands)
        return resolved, commands

    def _public_resolved_change_intent(self, intent: ResolvedChangeIntent) -> ResolvedChangeIntent:
        effect_cells = (
            []
            if intent.center_world_position is None
            else self._disk_world_cells_raw(
                tuple(int(value) for value in intent.center_world_position),
                int(intent.effective_radius),
            )
        )
        effect_bounds = self._buffer_cell_bounds(effect_cells)
        generated_commands = [self._public_world_command(command) for command in intent.generated_commands]
        if intent.center_world_position is not None:
            center_world_x = int(intent.center_world_position[0])
            center_world_y = int(intent.center_world_position[1])
            for command in generated_commands:
                x_field, y_field = TARGETED_COMMAND_COORD_FIELDS.get(command.kind, (None, None))
                if x_field is not None and y_field is not None and x_field in command.payload and y_field in command.payload:
                    command.payload[x_field] = center_world_x
                    command.payload[y_field] = center_world_y
        return replace(
            intent,
            center_position=(
                None
                if intent.center_world_position is None
                else tuple(int(value) for value in intent.center_world_position)
            ),
            effect_cells=effect_cells,
            effect_bounds=effect_bounds,
            generated_commands=generated_commands,
        )

    def _public_resolved_target(self, target: ResolvedTarget) -> ResolvedTarget:
        return replace(
            target,
            source_position=(
                None
                if target.source_world_position is None
                else tuple(int(value) for value in target.source_world_position)
            ),
            anchor_position=(
                None
                if target.anchor_world_position is None
                else tuple(int(value) for value in target.anchor_world_position)
            ),
            resolved_position=(
                None
                if target.resolved_world_position is None
                else tuple(int(value) for value in target.resolved_world_position)
            ),
        )

    def _resolve_carrier_intents(
        self,
        intents: list[CarrierIntent],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[dict[str, ResolvedCarrierIntent], list[WorldCommand]]:
        resolved: dict[str, ResolvedCarrierIntent] = {}
        commands: list[WorldCommand] = []
        for intent in intents:
            resolved_intent = self._resolve_carrier_intent(intent, resolved_targets)
            resolved[intent.intent_id] = resolved_intent
            commands.extend(WorldCommand(kind=command.kind, payload=deepcopy(command.payload)) for command in resolved_intent.generated_commands)
        return resolved, commands

    def _public_resolved_carrier_intent(self, intent: ResolvedCarrierIntent) -> ResolvedCarrierIntent:
        effect_cells: list[tuple[int, int]]
        if intent.effect_shape == "beam" and intent.source_world_position is not None and intent.impact_world_position is not None:
            effect_cells = self._capsule_world_cells_raw(
                tuple(int(value) for value in intent.source_world_position),
                tuple(int(value) for value in intent.impact_world_position),
                int(intent.effective_radius),
            )
        elif intent.impact_world_position is not None:
            effect_cells = self._disk_world_cells_raw(
                tuple(int(value) for value in intent.impact_world_position),
                int(intent.effective_radius),
            )
        else:
            effect_cells = [self._buffer_to_world_position(cell) for cell in intent.effect_cells]
        effect_bounds = self._buffer_cell_bounds(effect_cells)
        generated_commands = [self._public_world_command(command) for command in intent.generated_commands]
        if intent.kind == "light":
            origin_world_position = (
                tuple(int(value) for value in intent.source_world_position)
                if intent.effect_shape == "beam" and intent.source_world_position is not None
                else None
                if intent.impact_world_position is None
                else tuple(int(value) for value in intent.impact_world_position)
            )
            if origin_world_position is not None:
                for command in generated_commands:
                    if command.kind == "inject_light":
                        command.payload["x"] = int(origin_world_position[0])
                        command.payload["y"] = int(origin_world_position[1])
        elif intent.kind == "force" and intent.source_world_position is not None:
            origin_world_position = tuple(int(value) for value in intent.source_world_position)
            for command in generated_commands:
                if command.kind == "inject_force":
                    command.payload["x"] = int(origin_world_position[0])
                    command.payload["y"] = int(origin_world_position[1])
        elif intent.kind in {"material", "gas"}:
            world_cells = (
                effect_cells
                if intent.effect_shape == "beam"
                else []
                if intent.impact_world_position is None
                else [tuple(int(value) for value in intent.impact_world_position)]
            )
            target_kind = "inject_material" if intent.kind == "material" else "inject_gas"
            rewritten = iter(world_cells)
            for command in generated_commands:
                if command.kind != target_kind:
                    continue
                try:
                    world_cell = next(rewritten)
                except StopIteration:
                    break
                command.payload["x"] = int(world_cell[0])
                command.payload["y"] = int(world_cell[1])
        return replace(
            intent,
            source_position=(
                None
                if intent.source_world_position is None
                else tuple(int(value) for value in intent.source_world_position)
            ),
            impact_position=(
                None
                if intent.impact_world_position is None
                else tuple(int(value) for value in intent.impact_world_position)
            ),
            effect_cells=effect_cells,
            effect_bounds=effect_bounds,
            generated_commands=generated_commands,
        )

    def _resolve_carrier_intent(
        self,
        intent: CarrierIntent,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ResolvedCarrierIntent:
        fallback_mode = str(intent.fallback_mode or "nearest_empty").lower()
        kind = intent.kind.lower()
        if kind not in {"material", "gas", "light", "force"}:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="invalid_kind",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=float(intent.potency),
                stability=float(intent.stability),
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                note="unsupported carrier intent kind",
            )

        impact_world_position = self._resolve_intent_world_position(
            target_query_id=intent.target_query_id,
            center_x=intent.center_x,
            center_y=intent.center_y,
            target_dx=intent.target_dx,
            target_dy=intent.target_dy,
            resolved_targets=resolved_targets,
        )
        if impact_world_position is None:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="missing_target",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=float(intent.potency),
                stability=float(intent.stability),
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                note="carrier intent target could not be resolved",
            )

        potency = max(0.0, float(intent.potency))
        stability = min(1.0, max(0.0, float(intent.stability)))
        effective_radius = max(0, int(round(int(intent.radius) * potency)))
        attempted_world_position = self._apply_change_stability_drift(
            intent.intent_id,
            impact_world_position,
            effective_radius=effective_radius,
            stability=stability,
        )
        source_position: tuple[int, int] | None = None
        source_world_position: tuple[int, int] | None = None
        if (
            intent.source_entity_id is not None
            or intent.source_x is not None
            or intent.source_y is not None
            or kind in {"light", "force"}
            or intent.release_mode != "impact"
            or (intent.require_empty and fallback_mode == "source")
        ):
            source_position, source_world_position = self._resolve_intent_source_positions(
                source_entity_id=intent.source_entity_id,
                source_x=intent.source_x,
                source_y=intent.source_y,
            )

        drifted = attempted_world_position != impact_world_position
        drift_note = "low stability introduced deterministic impact drift" if drifted else None
        final_world_position, fallback_applied, fallback_note = self._resolve_legal_world_position(
            attempted_world_position,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            fallback_radius=int(intent.fallback_radius),
            effective_radius=effective_radius,
            source_world_position=source_world_position,
        )
        attempted_position = self._world_to_buffer_clamped(*attempted_world_position)
        if final_world_position is None:
            return ResolvedCarrierIntent(
                intent_id=intent.intent_id,
                status="blocked",
                kind=kind,
                target_query_id=intent.target_query_id,
                label=intent.label,
                release_mode=intent.release_mode,
                potency=potency,
                stability=stability,
                source_position=source_position,
                source_world_position=source_world_position,
                impact_position=attempted_position,
                impact_world_position=attempted_world_position,
                effective_radius=effective_radius,
                material=intent.material,
                gas_species=intent.gas_species,
                gas_amount=float(intent.gas_amount) * potency if kind == "gas" else 0.0,
                light_type=intent.light_type,
                light_strength=float(intent.light_strength) * potency if kind == "light" else 0.0,
                light_spread=float(intent.light_spread),
                force_radius=float(intent.force_radius) * potency if kind == "force" else 0.0,
                force_strength=float(intent.force_strength) * potency if kind == "force" else 0.0,
                force_lifetime=float(intent.force_lifetime),
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                fallback_applied=False,
                note=self._combine_resolution_notes(drift_note, fallback_note),
            )

        impact_position = self._world_to_buffer_clamped(*final_world_position)

        direction = None
        if source_world_position is not None:
            direction = self._normalized_world_direction(source_world_position, final_world_position)

        effect_shape = "impact"
        if intent.release_mode in {"beam", "projectile"} and source_world_position is not None:
            effect_shape = "beam"
            effect_world_cells = self._capsule_world_cells(source_world_position, final_world_position, effective_radius)
            raw_effect_world_cells = self._capsule_world_cells_raw(source_world_position, final_world_position, effective_radius)
        else:
            effect_world_cells = self._disk_world_cells(final_world_position, effective_radius)
            raw_effect_world_cells = self._disk_world_cells_raw(final_world_position, effective_radius)
        effect_cells = [self._world_to_buffer_clamped(world_x, world_y) for world_x, world_y in effect_world_cells]

        generated_commands: list[WorldCommand] = []
        base_meta: dict[str, Any] = {"resolved_carrier_intent_id": intent.intent_id}
        if intent.target_query_id is not None:
            base_meta["resolved_target_query_id"] = intent.target_query_id

        note = self._combine_resolution_notes(drift_note, fallback_note)
        status = self._intent_resolution_status(drifted=drifted, fallback_applied=fallback_applied)

        resolved_force_radius = float(intent.force_radius) * potency if intent.force_radius > 0.0 else max(1.0, float(effective_radius))

        if kind == "material":
            if intent.material is None:
                return ResolvedCarrierIntent(
                    intent_id=intent.intent_id,
                    status="invalid_payload",
                    kind=kind,
                    target_query_id=intent.target_query_id,
                    label=intent.label,
                    release_mode=intent.release_mode,
                    potency=potency,
                    stability=stability,
                    source_position=source_position,
                    source_world_position=source_world_position,
                    impact_position=impact_position,
                    impact_world_position=final_world_position,
                    effective_radius=effective_radius,
                    require_empty=bool(intent.require_empty),
                    fallback_mode=fallback_mode,
                    fallback_applied=fallback_applied,
                    note="material carrier intent requires material",
                )
            generated_commands.extend(
                WorldCommand(
                    kind="inject_material",
                    payload={
                        "x": int(cell[0]),
                        "y": int(cell[1]),
                        "material": intent.material,
                        "radius": 0 if effect_shape == "beam" else effective_radius,
                        **base_meta,
                    },
                )
                for cell in (raw_effect_world_cells if effect_shape == "beam" else [final_world_position])
            )
        elif kind == "gas":
            if intent.gas_species is None or intent.gas_amount == 0.0:
                return ResolvedCarrierIntent(
                    intent_id=intent.intent_id,
                    status="invalid_payload",
                    kind=kind,
                    target_query_id=intent.target_query_id,
                    label=intent.label,
                    release_mode=intent.release_mode,
                    potency=potency,
                    stability=stability,
                    source_position=source_position,
                    source_world_position=source_world_position,
                    impact_position=impact_position,
                    impact_world_position=final_world_position,
                    effective_radius=effective_radius,
                    require_empty=bool(intent.require_empty),
                    fallback_mode=fallback_mode,
                    fallback_applied=fallback_applied,
                    note="gas carrier intent requires gas_species and non-zero gas_amount",
                )
            scaled_amount = float(intent.gas_amount) * potency
            gas_cells = raw_effect_world_cells if effect_shape == "beam" else [final_world_position]
            per_cell_amount = scaled_amount / max(1, len(gas_cells))
            generated_commands.extend(
                WorldCommand(
                    kind="inject_gas",
                    payload={
                        "x": int(cell[0]),
                        "y": int(cell[1]),
                        "species": intent.gas_species,
                        "amount": per_cell_amount,
                        "radius": 0 if effect_shape == "beam" else effective_radius,
                        **base_meta,
                    },
                )
                for cell in gas_cells
            )
        elif kind == "light":
            if intent.light_type is None:
                return ResolvedCarrierIntent(
                    intent_id=intent.intent_id,
                    status="invalid_payload",
                    kind=kind,
                    target_query_id=intent.target_query_id,
                    label=intent.label,
                    release_mode=intent.release_mode,
                    potency=potency,
                    stability=stability,
                    source_position=source_position,
                    source_world_position=source_world_position,
                    impact_position=impact_position,
                    impact_world_position=final_world_position,
                    effective_radius=effective_radius,
                    require_empty=bool(intent.require_empty),
                    fallback_mode=fallback_mode,
                    fallback_applied=fallback_applied,
                    note="light carrier intent requires light_type",
                )
            light_strength = float(intent.light_strength) * potency
            origin_position = impact_position
            light_direction = (0.0, 0.0)
            light_range = max(1, effective_radius if effective_radius > 0 else int(intent.radius) or 1)
            if intent.release_mode in {"beam", "projectile"} and source_position is not None and source_world_position is not None:
                origin_position = source_position
                light_direction = (0.0, 0.0) if direction is None else direction
                if int(intent.radius) <= 0:
                    light_range = max(
                        1,
                        int(round(np.linalg.norm(np.asarray(final_world_position, dtype=np.float32) - np.asarray(source_world_position, dtype=np.float32)))),
                    )
            generated_commands.append(
                WorldCommand(
                    kind="inject_light",
                    payload={
                        "x": int(source_world_position[0] if intent.release_mode in {"beam", "projectile"} and source_world_position is not None else final_world_position[0]),
                        "y": int(source_world_position[1] if intent.release_mode in {"beam", "projectile"} and source_world_position is not None else final_world_position[1]),
                        "light_type": intent.light_type,
                        "strength": light_strength,
                        "radius": light_range,
                        "direction": light_direction,
                        "spread": float(intent.light_spread),
                        **base_meta,
                    },
                )
            )
        else:
            if intent.force_strength == 0.0 and intent.force_radius == 0.0:
                return ResolvedCarrierIntent(
                    intent_id=intent.intent_id,
                    status="invalid_payload",
                    kind=kind,
                    target_query_id=intent.target_query_id,
                    label=intent.label,
                    release_mode=intent.release_mode,
                    potency=potency,
                    stability=stability,
                    source_position=source_position,
                    source_world_position=source_world_position,
                    impact_position=impact_position,
                    impact_world_position=final_world_position,
                    effective_radius=effective_radius,
                    require_empty=bool(intent.require_empty),
                    fallback_mode=fallback_mode,
                    fallback_applied=fallback_applied,
                    note="force carrier intent requires non-zero force radius or strength",
                )
            if source_position is None or source_world_position is None:
                source_position, source_world_position = self._resolve_intent_source_positions(
                    source_entity_id=None,
                    source_x=None,
                    source_y=None,
                )
                direction = self._normalized_world_direction(source_world_position, final_world_position)
            force_direction = (1.0, 0.0) if direction is None else direction
            generated_commands.append(
                WorldCommand(
                    kind="inject_force",
                    payload={
                        "x": int(source_world_position[0]),
                        "y": int(source_world_position[1]),
                        "direction": force_direction,
                        "radius": resolved_force_radius,
                        "strength": float(intent.force_strength) * potency,
                        "lifetime": float(intent.force_lifetime),
                        **base_meta,
                    },
                )
            )

        return ResolvedCarrierIntent(
            intent_id=intent.intent_id,
            status=status,
            kind=kind,
            target_query_id=intent.target_query_id,
            label=intent.label,
            release_mode=intent.release_mode,
            potency=potency,
            stability=stability,
            source_position=source_position,
            source_world_position=source_world_position,
            impact_position=impact_position,
            impact_world_position=final_world_position,
            effective_radius=effective_radius,
            material=intent.material,
            gas_species=intent.gas_species,
            gas_amount=float(intent.gas_amount) * potency if kind == "gas" else 0.0,
            light_type=intent.light_type,
            light_strength=float(intent.light_strength) * potency if kind == "light" else 0.0,
            light_spread=float(intent.light_spread),
            force_radius=resolved_force_radius if kind == "force" else 0.0,
            force_strength=float(intent.force_strength) * potency if kind == "force" else 0.0,
            force_lifetime=float(intent.force_lifetime),
            direction=direction,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            fallback_applied=fallback_applied,
            effect_shape=effect_shape,
            effect_cells=effect_cells,
            effect_bounds=self._buffer_cell_bounds(effect_cells),
            generated_commands=generated_commands,
            note=note,
        )

    def _resolve_change_intent(
        self,
        intent: ChangeIntent,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> ResolvedChangeIntent:
        fallback_mode = str(intent.fallback_mode or "nearest_empty").lower()
        if intent.material is None and not intent.temperature_delta and intent.velocity is None:
            return ResolvedChangeIntent(
                intent_id=intent.intent_id,
                status="empty",
                target_query_id=intent.target_query_id,
                label=intent.label,
                potency=float(intent.potency),
                stability=float(intent.stability),
                velocity_carrier=intent.velocity_carrier,
                velocity_mode=intent.velocity_mode,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                note="change intent had no material, temperature, or velocity edit",
            )

        world_position = self._resolve_change_intent_world_position(intent, resolved_targets)
        if world_position is None:
            return ResolvedChangeIntent(
                intent_id=intent.intent_id,
                status="missing_target",
                target_query_id=intent.target_query_id,
                label=intent.label,
                potency=float(intent.potency),
                stability=float(intent.stability),
                velocity_carrier=intent.velocity_carrier,
                velocity_mode=intent.velocity_mode,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                note="change intent target could not be resolved",
            )

        potency = max(0.0, float(intent.potency))
        stability = min(1.0, max(0.0, float(intent.stability)))
        effective_radius = max(0, int(round(int(intent.radius) * potency)))
        attempted_world_position = self._apply_change_stability_drift(
            intent.intent_id,
            world_position,
            effective_radius=effective_radius,
            stability=stability,
        )
        scaled_temperature_delta = float(intent.temperature_delta) * potency
        scaled_velocity = None
        if intent.velocity is not None:
            scaled_velocity = (
                float(intent.velocity[0]) * potency,
                float(intent.velocity[1]) * potency,
            )

        drifted = attempted_world_position != world_position
        drift_note = "low stability introduced deterministic target drift" if drifted else None
        final_world_position, fallback_applied, fallback_note = self._resolve_legal_world_position(
            attempted_world_position,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            fallback_radius=int(intent.fallback_radius),
            effective_radius=effective_radius,
            source_world_position=None,
        )
        attempted_position = self._world_to_buffer_clamped(*attempted_world_position)
        if final_world_position is None:
            return ResolvedChangeIntent(
                intent_id=intent.intent_id,
                status="blocked",
                target_query_id=intent.target_query_id,
                label=intent.label,
                potency=potency,
                stability=stability,
                center_position=attempted_position,
                center_world_position=attempted_world_position,
                effective_radius=effective_radius,
                material=intent.material,
                temperature_delta=scaled_temperature_delta,
                velocity=scaled_velocity,
                velocity_carrier=intent.velocity_carrier,
                velocity_mode=intent.velocity_mode,
                require_empty=bool(intent.require_empty),
                fallback_mode=fallback_mode,
                fallback_applied=False,
                note=self._combine_resolution_notes(drift_note, fallback_note),
            )

        center_position = self._world_to_buffer_clamped(*final_world_position)
        effect_world_cells = self._disk_world_cells(final_world_position, effective_radius)
        effect_cells = [self._world_to_buffer_clamped(world_x, world_y) for world_x, world_y in effect_world_cells]

        generated_commands: list[WorldCommand] = []
        base_meta: dict[str, Any] = {"resolved_change_intent_id": intent.intent_id}
        if intent.target_query_id is not None:
            base_meta["resolved_target_query_id"] = intent.target_query_id
        if intent.material is not None:
            generated_commands.append(
                WorldCommand(
                    kind="inject_material",
                    payload={
                        "x": int(final_world_position[0]),
                        "y": int(final_world_position[1]),
                        "material": intent.material,
                        "radius": effective_radius,
                        **base_meta,
                    },
                )
            )
        if scaled_temperature_delta != 0.0:
            generated_commands.append(
                WorldCommand(
                    kind="inject_temperature",
                    payload={
                        "x": int(final_world_position[0]),
                        "y": int(final_world_position[1]),
                        "delta": scaled_temperature_delta,
                        "radius": effective_radius,
                        **base_meta,
                    },
                )
            )
        if scaled_velocity is not None:
            generated_commands.append(
                WorldCommand(
                    kind="inject_velocity",
                    payload={
                        "x": int(final_world_position[0]),
                        "y": int(final_world_position[1]),
                        "velocity": [float(scaled_velocity[0]), float(scaled_velocity[1])],
                        "radius": effective_radius,
                        "carrier": intent.velocity_carrier,
                        "mode": intent.velocity_mode,
                        **base_meta,
                    },
                )
            )

        status = self._intent_resolution_status(drifted=drifted, fallback_applied=fallback_applied)
        note = self._combine_resolution_notes(drift_note, fallback_note)

        return ResolvedChangeIntent(
            intent_id=intent.intent_id,
            status=status,
            target_query_id=intent.target_query_id,
            label=intent.label,
            potency=potency,
            stability=stability,
            center_position=center_position,
            center_world_position=final_world_position,
            effective_radius=effective_radius,
            material=intent.material,
            temperature_delta=scaled_temperature_delta,
            velocity=scaled_velocity,
            velocity_carrier=intent.velocity_carrier,
            velocity_mode=intent.velocity_mode,
            require_empty=bool(intent.require_empty),
            fallback_mode=fallback_mode,
            fallback_applied=fallback_applied,
            effect_shape="burst",
            effect_cells=effect_cells,
            effect_bounds=self._buffer_cell_bounds(effect_cells),
            generated_commands=generated_commands,
            note=note,
        )

    def _resolve_change_intent_world_position(
        self,
        intent: ChangeIntent,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[int, int] | None:
        return self._resolve_intent_world_position(
            target_query_id=intent.target_query_id,
            center_x=intent.center_x,
            center_y=intent.center_y,
            target_dx=intent.target_dx,
            target_dy=intent.target_dy,
            resolved_targets=resolved_targets,
        )

    def _resolve_intent_world_position(
        self,
        *,
        target_query_id: str | None,
        center_x: int | None,
        center_y: int | None,
        target_dx: int,
        target_dy: int,
        resolved_targets: dict[str, ResolvedTarget],
    ) -> tuple[int, int] | None:
        if target_query_id is not None:
            target = resolved_targets.get(target_query_id)
            if target is None or target.status not in {"resolved", "drifted"} or target.resolved_world_position is None:
                return None
            return (
                int(target.resolved_world_position[0]) + int(target_dx),
                int(target.resolved_world_position[1]) + int(target_dy),
            )
        if center_x is None or center_y is None:
            return None
        return (
            int(center_x) + int(target_dx),
            int(center_y) + int(target_dy),
        )

    def _resolve_intent_source_positions(
        self,
        *,
        source_entity_id: int | None,
        source_x: int | None,
        source_y: int | None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if source_entity_id is not None:
            entity = self.entity_states.get(int(source_entity_id))
            if entity is not None:
                world_position = self._entity_center_world_position(entity)
                return self._world_to_buffer_clamped(*world_position), world_position
        if source_x is not None and source_y is not None:
            world_position = (int(source_x), int(source_y))
            buffer_position = self._world_to_buffer_clamped(*world_position)
            return buffer_position, world_position
        buffer_position = self._default_target_source_position()
        return buffer_position, self._buffer_to_world_position(buffer_position)

    @staticmethod
    def _normalized_world_direction(
        source_world_position: tuple[int, int],
        target_world_position: tuple[int, int],
    ) -> tuple[float, float] | None:
        delta = np.asarray(target_world_position, dtype=np.float32) - np.asarray(source_world_position, dtype=np.float32)
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            return None
        direction = delta / length
        return (float(direction[0]), float(direction[1]))

    def _disk_world_cells(self, center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        radius = max(0, int(radius))
        cx, cy = center_world_position
        cells: list[tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                cells.append(self._clamp_world_position(cx + dx, cy + dy))
        if not cells:
            cells.append(self._clamp_world_position(cx, cy))
        return sorted(set(cells))

    @staticmethod
    def _disk_world_cells_raw(center_world_position: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        radius = max(0, int(radius))
        cx, cy = center_world_position
        cells: list[tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                cells.append((int(cx + dx), int(cy + dy)))
        if not cells:
            cells.append((int(cx), int(cy)))
        return sorted(set(cells))

    def _line_world_cells(
        self,
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        x0, y0 = (int(value) for value in start_world_position)
        x1, y1 = (int(value) for value in end_world_position)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cells: list[tuple[int, int]] = []
        while True:
            cells.append(self._clamp_world_position(x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return cells

    @staticmethod
    def _line_world_cells_raw(
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        x0, y0 = (int(value) for value in start_world_position)
        x1, y1 = (int(value) for value in end_world_position)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cells: list[tuple[int, int]] = []
        while True:
            cells.append((int(x0), int(y0)))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return cells

    def _capsule_world_cells(
        self,
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
        radius: int,
    ) -> list[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for world_position in self._line_world_cells(start_world_position, end_world_position):
            cells.update(self._disk_world_cells(world_position, radius))
        return sorted(cells)

    def _capsule_world_cells_raw(
        self,
        start_world_position: tuple[int, int],
        end_world_position: tuple[int, int],
        radius: int,
    ) -> list[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for world_position in self._line_world_cells_raw(start_world_position, end_world_position):
            cells.update(self._disk_world_cells_raw(world_position, radius))
        return sorted(cells)

    @staticmethod
    def _buffer_cell_bounds(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        if not cells:
            return None
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

    def _apply_change_stability_drift(
        self,
        intent_id: str,
        world_position: tuple[int, int],
        *,
        effective_radius: int,
        stability: float,
    ) -> tuple[int, int]:
        drift_radius = int(round((1.0 - stability) * max(1, effective_radius + 1)))
        if drift_radius <= 0:
            return world_position
        seed = zlib.crc32(intent_id.encode("utf-8")) & 0xFFFFFFFF
        span = drift_radius * 2 + 1
        dx = int(seed % span) - drift_radius
        dy = int((seed // span) % span) - drift_radius
        return self._clamp_world_position(world_position[0] + dx, world_position[1] + dy)

    def _resolve_legal_world_position(
        self,
        world_position: tuple[int, int],
        *,
        require_empty: bool,
        fallback_mode: str,
        fallback_radius: int,
        effective_radius: int,
        source_world_position: tuple[int, int] | None,
    ) -> tuple[tuple[int, int] | None, bool, str | None]:
        if not require_empty:
            return world_position, False, None

        clamped_world_position = self._clamp_world_position(*world_position)
        if self._world_cell_is_empty(*clamped_world_position):
            return clamped_world_position, False, None

        if fallback_mode == "nearest_empty":
            search_radius = max(0, int(fallback_radius))
            if search_radius <= 0:
                search_radius = max(1, int(effective_radius) + 1)
            empty_world_position = self._find_nearest_empty_world_position(
                clamped_world_position,
                radius=search_radius,
            )
            if empty_world_position is not None:
                return empty_world_position, True, "occupied target fell back to nearest empty cell"
            return None, False, "occupied target had no empty fallback cell"

        if fallback_mode == "source":
            if source_world_position is None:
                return None, False, "occupied target requested source fallback without a source position"
            fallback_world_position = self._clamp_world_position(*source_world_position)
            if self._world_cell_is_empty(*fallback_world_position):
                return fallback_world_position, True, "occupied target fell back to source cell"
            return None, False, "occupied target could not fall back to the source cell"

        return None, False, f"occupied target requested unsupported fallback mode '{fallback_mode}'"

    @staticmethod
    def _intent_resolution_status(*, drifted: bool, fallback_applied: bool) -> str:
        if fallback_applied:
            return "fallback"
        if drifted:
            return "drifted"
        return "resolved"

    @staticmethod
    def _combine_resolution_notes(*notes: str | None) -> str | None:
        filtered = [note for note in notes if note]
        if not filtered:
            return None
        return "; ".join(filtered)

    def _resolve_targeted_commands(
        self,
        commands: list[WorldCommand],
        resolved_targets: dict[str, ResolvedTarget],
    ) -> list[WorldCommand]:
        resolved_commands: list[WorldCommand] = []
        for command in commands:
            payload = deepcopy(command.payload)
            target_query_id = payload.pop("target_query_id", None)
            target_dx = int(payload.pop("target_dx", 0))
            target_dy = int(payload.pop("target_dy", 0))
            if target_query_id is None:
                resolved_commands.append(WorldCommand(kind=command.kind, payload=payload))
                continue
            target = resolved_targets.get(str(target_query_id))
            fields = TARGETED_COMMAND_COORD_FIELDS.get(command.kind)
            if target is None or target.status != "resolved" or target.resolved_world_position is None or fields is None:
                continue
            world_x = int(target.resolved_world_position[0]) + target_dx
            world_y = int(target.resolved_world_position[1]) + target_dy
            x_field, y_field = fields
            if command.kind in {"request_readback", "advance_paging"}:
                payload[x_field] = int(world_x)
                payload[y_field] = int(world_y)
                payload["target_query_id"] = str(target_query_id)
                payload["target_dx"] = int(target_dx)
                payload["target_dy"] = int(target_dy)
            else:
                payload[x_field] = int(world_x)
                payload[y_field] = int(world_y)
                payload["resolved_target_query_id"] = str(target_query_id)
            resolved_commands.append(WorldCommand(kind=command.kind, payload=payload))
        return resolved_commands

    def _resolve_target_query(self, query: TargetQuery) -> ResolvedTarget:
        resolved_distance_cells = self._resolve_target_query_distance_cells(query)
        source_position = self._resolve_query_source_position(query)
        if source_position is None:
            return ResolvedTarget(
                query_id=query.query_id,
                status="missing_source",
                anchor_filters=query.anchor_filters,
                direction=query.direction,
                distance_cells=resolved_distance_cells,
                distance_meters=query.distance_meters,
                distance_hint=query.distance_hint,
                label=query.label,
                note="source position could not be resolved",
            )

        source_world_position = self._buffer_to_world_position(source_position)
        if query.source_entity_id is not None:
            entity = self.entity_states.get(int(query.source_entity_id))
            if entity is not None:
                source_world_position = self._entity_center_world_position(entity)
        anchor = self._resolve_anchor_target(query, source_world_position)
        if anchor is None:
            return ResolvedTarget(
                query_id=query.query_id,
                status="missing_anchor",
                anchor_filters=query.anchor_filters,
                direction=query.direction,
                distance_cells=resolved_distance_cells,
                distance_meters=query.distance_meters,
                distance_hint=query.distance_hint,
                label=query.label,
                source_position=source_position,
                source_world_position=source_world_position,
                note="no matching anchor was found in the loaded world window",
            )

        resolved_world_position = anchor["world_position"]
        direction_vector = self._query_direction_vector(query, source_entity_id=query.source_entity_id)
        if direction_vector is not None and resolved_distance_cells:
            resolved_world_position = self._clamp_world_position(
                resolved_world_position[0] + direction_vector[0] * resolved_distance_cells,
                resolved_world_position[1] + direction_vector[1] * resolved_distance_cells,
            )

        if query.require_empty:
            empty_world_position = self._find_nearest_empty_world_position(
                resolved_world_position,
                radius=max(0, int(query.search_radius)),
            )
            if empty_world_position is None:
                return ResolvedTarget(
                    query_id=query.query_id,
                    status="blocked",
                    anchor_filters=query.anchor_filters,
                    direction=query.direction,
                    distance_cells=resolved_distance_cells,
                    distance_meters=query.distance_meters,
                    distance_hint=query.distance_hint,
                    label=query.label,
                    source_position=source_position,
                    source_world_position=source_world_position,
                    anchor_kind=str(anchor["kind"]),
                    anchor_entity_id=anchor["entity_id"],
                    anchor_position=anchor["buffer_position"],
                    anchor_world_position=anchor["world_position"],
                    note="no empty landing cell was found near the resolved target",
                )
            resolved_world_position = empty_world_position

        resolved_position = self._world_to_buffer_clamped(*resolved_world_position)
        return ResolvedTarget(
            query_id=query.query_id,
            status="resolved",
            anchor_filters=query.anchor_filters,
            direction=query.direction,
            distance_cells=resolved_distance_cells,
            distance_meters=query.distance_meters,
            distance_hint=query.distance_hint,
            label=query.label,
            source_position=source_position,
            source_world_position=source_world_position,
            anchor_kind=str(anchor["kind"]),
            anchor_entity_id=anchor["entity_id"],
            anchor_position=anchor["buffer_position"],
            anchor_world_position=anchor["world_position"],
            resolved_position=resolved_position,
            resolved_world_position=resolved_world_position,
        )

    def _resolve_target_query_distance_cells(self, query: TargetQuery) -> int:
        if int(query.distance_cells) != 0:
            return int(query.distance_cells)
        if query.distance_meters is not None:
            return self._distance_meters_to_cells(float(query.distance_meters))
        if query.distance_hint is not None:
            return int(TARGET_QUERY_DISTANCE_HINT_CELLS.get(str(query.distance_hint).lower(), 0))
        return 0

    @staticmethod
    def _distance_meters_to_cells(distance_meters: float) -> int:
        cells = int(round(float(distance_meters) * TARGET_QUERY_CELLS_PER_METER))
        if cells == 0 and abs(float(distance_meters)) > 1e-6:
            return 1 if distance_meters > 0.0 else -1
        return cells

    def _resolve_query_source_position(self, query: TargetQuery) -> tuple[int, int] | None:
        if query.source_entity_id is not None:
            entity = self.entity_states.get(int(query.source_entity_id))
            if entity is None:
                return None
            return self._world_to_buffer_clamped(*self._entity_center_world_position(entity))
        if query.source_x is not None and query.source_y is not None:
            return self._world_to_buffer_clamped(int(query.source_x), int(query.source_y))
        if query.source_x is None and query.source_y is None:
            return self._default_target_source_position()
        return None

    def _default_target_source_position(self) -> tuple[int, int]:
        return (
            (int(self.paging.buffer_origin_x) + int(self.paging.active_width) // 2) % self.width,
            (int(self.paging.buffer_origin_y) + int(self.paging.active_height) // 2) % self.height,
        )

    def _resolve_anchor_target(
        self,
        query: TargetQuery,
        source_world_position: tuple[int, int],
    ) -> dict[str, Any] | None:
        directional_filters = [item for item in query.anchor_filters if item in CARDINAL_DIRECTION_VECTORS or item in {"forward", "backward"}]
        terrain_filters = [item for item in query.anchor_filters if item in TERRAIN_ANCHOR_FILTERS]
        entity_filters = [
            item
            for item in query.anchor_filters
            if item not in TERRAIN_ANCHOR_FILTERS
            and item not in IGNORED_ANCHOR_FILTERS
            and item not in CARDINAL_DIRECTION_VECTORS
            and item not in {"forward", "backward"}
        ]
        entity_anchor = None
        terrain_anchor = None
        if query.anchor_entity_id is not None or entity_filters or (directional_filters and not terrain_filters):
            entity_anchor = self._resolve_entity_anchor(
                query,
                source_world_position,
                direction_filter=directional_filters[0] if directional_filters else None,
            )
        if entity_anchor is not None:
            return entity_anchor
        if terrain_filters:
            terrain_anchor = self._resolve_terrain_anchor(
                source_world_position,
                terrain_filters,
                direction_filter=directional_filters[0] if directional_filters else None,
            )
        if terrain_anchor is not None:
            return terrain_anchor
        if query.anchor_entity_id is None and not query.anchor_filters:
            return {
                "kind": "source",
                "entity_id": query.source_entity_id,
                "buffer_position": self._world_to_buffer_clamped(*source_world_position),
                "world_position": source_world_position,
            }
        return None

    def _resolve_entity_anchor(
        self,
        query: TargetQuery,
        source_world_position: tuple[int, int],
        *,
        direction_filter: str | None,
    ) -> dict[str, Any] | None:
        best: tuple[float, int, tuple[int, int], tuple[int, int]] | None = None
        for entity in self.entity_states.values():
            if query.anchor_entity_id is not None:
                if entity.entity_id != int(query.anchor_entity_id):
                    continue
            else:
                if query.source_entity_id is not None and entity.entity_id == int(query.source_entity_id):
                    continue
                if not self._entity_matches_anchor_filters(entity, query.anchor_filters):
                    continue
            world_position = self._entity_center_world_position(entity)
            buffer_position = self._world_to_buffer_clamped(*world_position)
            if direction_filter is not None and not self._matches_direction_filter(
                source_world_position,
                world_position,
                direction_filter,
                source_entity_id=query.source_entity_id,
            ):
                continue
            distance_sq = self._world_distance_sq(source_world_position, world_position)
            candidate = (distance_sq, int(entity.entity_id), buffer_position, world_position)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        _, entity_id, buffer_position, world_position = best
        return {
            "kind": "entity",
            "entity_id": entity_id,
            "buffer_position": buffer_position,
            "world_position": world_position,
        }

    def _resolve_terrain_anchor(
        self,
        source_world_position: tuple[int, int],
        terrain_filters: list[str],
        *,
        direction_filter: str | None,
    ) -> dict[str, Any] | None:
        for terrain_filter in terrain_filters:
            best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
            for y in range(self.height):
                for x in range(self.width):
                    if not self._terrain_cell_matches(x, y, terrain_filter):
                        continue
                    world_position = self._buffer_to_world_position((x, y))
                    if direction_filter is not None and not self._matches_direction_filter(
                        source_world_position,
                        world_position,
                        direction_filter,
                        source_entity_id=None,
                    ):
                        continue
                    distance_sq = self._world_distance_sq(source_world_position, world_position)
                    candidate = (distance_sq, (x, y), world_position)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                _, buffer_position, world_position = best
                return {
                    "kind": "terrain",
                    "entity_id": None,
                    "buffer_position": buffer_position,
                    "world_position": world_position,
                }
        return None

    def _entity_matches_anchor_filters(self, entity: EntityState, filters: tuple[str, ...]) -> bool:
        area = max(1, int(entity.width) * int(entity.height))
        entity_tags = set(entity.tags)
        for item in filters:
            if item in TERRAIN_ANCHOR_FILTERS or item in IGNORED_ANCHOR_FILTERS:
                continue
            if item in CARDINAL_DIRECTION_VECTORS or item in {"forward", "backward"}:
                continue
            if item == "big":
                if area < 4:
                    return False
                continue
            if item == "small":
                if area > 1:
                    return False
                continue
            if item not in entity_tags:
                return False
        return True

    def _terrain_cell_matches(self, x: int, y: int, terrain_filter: str) -> bool:
        material_id, phase = self._material_state_for_position(
            x,
            y,
            blocked_cells=self._resolver_blocked_cells,
            released_cells=self._resolver_released_cells,
        )
        if terrain_filter == "empty":
            return material_id == 0
        if terrain_filter == "tree":
            return self._terrain_tree_cell_matches(x, y, material_id, phase)
        if terrain_filter in {"liquid", "pool"}:
            return material_id != 0 and phase == int(Phase.LIQUID)
        if terrain_filter == "solid":
            return material_id != 0 and phase != int(Phase.LIQUID)
        if terrain_filter == "hill":
            return self._terrain_hill_cell_matches(x, y, material_id, phase)
        if terrain_filter == "wall":
            if material_id == 0 or phase == int(Phase.LIQUID):
                return False
            above_material, above_phase = (0, 0) if y <= 0 else self._material_state_for_position(
                x,
                y - 1,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            below_material, below_phase = (0, 0) if y + 1 >= self.height else self._material_state_for_position(
                x,
                y + 1,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            left_material, _ = (0, 0) if x <= 0 else self._material_state_for_position(
                x - 1,
                y,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            right_material, _ = (0, 0) if x + 1 >= self.width else self._material_state_for_position(
                x + 1,
                y,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            vertical_neighbor = (
                (above_material != 0 and above_phase != int(Phase.LIQUID))
                or (below_material != 0 and below_phase != int(Phase.LIQUID))
            )
            horizontal_edge = (
                (left_material == 0)
                or (right_material == 0)
            )
            return vertical_neighbor and horizontal_edge
        if terrain_filter == "hole":
            if material_id != 0 or y + 1 >= self.height or x == 0 or x + 1 >= self.width:
                return False
            below_material, below_phase = self._material_state_for_position(
                x,
                y + 1,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            left_material, left_phase = self._material_state_for_position(
                x - 1,
                y,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            right_material, right_phase = self._material_state_for_position(
                x + 1,
                y,
                blocked_cells=self._resolver_blocked_cells,
                released_cells=self._resolver_released_cells,
            )
            below_solid = below_material != 0 and below_phase != int(Phase.LIQUID)
            left_solid = left_material != 0 and left_phase != int(Phase.LIQUID)
            right_solid = right_material != 0 and right_phase != int(Phase.LIQUID)
            return bool(below_solid and left_solid and right_solid)
        return False

    def _terrain_tree_cell_matches(self, x: int, y: int, material_id: int, phase: int) -> bool:
        if material_id == 0 or phase in {int(Phase.LIQUID), int(Phase.POWDER)}:
            return False
        if not self._world_cell_material_has_tag(x, y, "plant"):
            return False
        if not (
            self._world_cell_material_has_tag(x, y - 1, "plant")
            or self._world_cell_material_has_tag(x, y + 1, "plant")
        ):
            return False
        plant_neighbors = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self._world_cell_material_has_tag(x + dx, y + dy, "plant"):
                    plant_neighbors += 1
        return plant_neighbors >= 2

    def _terrain_hill_cell_matches(self, x: int, y: int, material_id: int, phase: int) -> bool:
        if material_id == 0 or phase == int(Phase.LIQUID):
            return False
        if self._world_cell_material_has_tag(x, y, "plant") or self._world_cell_material_has_tag(x, y, "placeholder"):
            return False
        if not self._world_cell_is_empty_local(x, y - 1):
            return False
        if self._world_cell_is_solid_local(x - 1, y) or self._world_cell_is_solid_local(x + 1, y):
            return False
        left_support = self._world_cell_is_solid_local(x - 1, y + 1) or self._world_cell_is_solid_local(x - 1, y + 2)
        right_support = self._world_cell_is_solid_local(x + 1, y + 1) or self._world_cell_is_solid_local(x + 1, y + 2)
        return left_support and right_support

    def _world_cell_is_solid_local(self, x: int, y: int) -> bool:
        material_id, phase = self._bounded_material_state_for_position(x, y)
        return material_id != 0 and phase != int(Phase.LIQUID)

    def _world_cell_is_empty_local(self, x: int, y: int) -> bool:
        material_id, _ = self._bounded_material_state_for_position(x, y)
        return material_id == 0

    def _world_cell_material_has_tag(self, x: int, y: int, tag: str) -> bool:
        material_id, _ = self._bounded_material_state_for_position(x, y)
        if material_id == 0:
            return False
        material_id = int(material_id)
        if tag == "plant":
            return self._shadow_material_is_plant(material_id)
        if tag == "placeholder":
            return self._shadow_material_is_placeholder(material_id)
        material = self._shadow_material_def(material_id)
        return material is not None and tag in material.tags

    def _bounded_material_state_for_position(self, x: int, y: int) -> tuple[int, int]:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return (0, 0)
        return self._material_state_for_position(
            x,
            y,
            blocked_cells=self._resolver_blocked_cells,
            released_cells=self._resolver_released_cells,
        )

    def _matches_direction_filter(
        self,
        source_world_position: tuple[int, int],
        candidate_world_position: tuple[int, int],
        direction_name: str,
        *,
        source_entity_id: int | None,
    ) -> bool:
        direction = self._direction_vector(direction_name, source_entity_id=source_entity_id)
        if direction is None:
            return True
        delta_x = candidate_world_position[0] - source_world_position[0]
        delta_y = candidate_world_position[1] - source_world_position[1]
        if direction[0] < 0:
            return delta_x < 0
        if direction[0] > 0:
            return delta_x > 0
        if direction[1] < 0:
            return delta_y < 0
        if direction[1] > 0:
            return delta_y > 0
        return True

    def _query_direction_vector(
        self,
        query: TargetQuery,
        *,
        source_entity_id: int | None,
    ) -> tuple[int, int] | None:
        if query.direction is None:
            return None
        return self._direction_vector(query.direction, source_entity_id=source_entity_id)

    def _direction_vector(
        self,
        direction_name: str,
        *,
        source_entity_id: int | None,
    ) -> tuple[int, int] | None:
        direction_key = direction_name.lower()
        if direction_key in CARDINAL_DIRECTION_VECTORS:
            return CARDINAL_DIRECTION_VECTORS[direction_key]
        if direction_key not in {"forward", "backward"}:
            return None
        facing_x, facing_y = self._source_facing_vector(source_entity_id)
        if abs(facing_x) >= abs(facing_y):
            direction = (1, 0) if facing_x >= 0.0 else (-1, 0)
        else:
            direction = (0, 1) if facing_y >= 0.0 else (0, -1)
        if direction_key == "backward":
            return (-direction[0], -direction[1])
        return direction

    def _source_facing_vector(self, source_entity_id: int | None) -> tuple[float, float]:
        if source_entity_id is not None:
            entity = self.entity_states.get(int(source_entity_id))
            if entity is not None:
                if entity.facing_xy is not None:
                    return (float(entity.facing_xy[0]), float(entity.facing_xy[1]))
                if entity.velocity_xy != (0.0, 0.0):
                    return (float(entity.velocity_xy[0]), float(entity.velocity_xy[1]))
        return (1.0, 0.0)

    def _entity_center_buffer_position(self, entity: EntityState) -> tuple[int, int]:
        return (
            int(entity.x) + max(0, int(entity.width) - 1) // 2,
            int(entity.y) + max(0, int(entity.height) - 1) // 2,
        )

    def _entity_center_world_position(self, entity: EntityState) -> tuple[int, int]:
        if entity.world_x is not None and entity.world_y is not None:
            return (
                int(entity.world_x) + max(0, int(entity.width) - 1) // 2,
                int(entity.world_y) + max(0, int(entity.height) - 1) // 2,
            )
        return self._buffer_to_world_position(self._entity_center_buffer_position(entity))

    def _buffer_to_world_position(self, position: tuple[int, int]) -> tuple[int, int]:
        world_x, world_y = self.paging.buffer_to_world(int(position[0]), int(position[1]))
        return (int(world_x), int(world_y))

    def _buffer_to_world_float_position(self, position: tuple[float, float]) -> tuple[float, float]:
        world_x = float(self.paging.origin_x) + ((float(position[0]) - float(self.paging.buffer_origin_x)) % float(self.width))
        world_y = float(self.paging.origin_y) + ((float(position[1]) - float(self.paging.buffer_origin_y)) % float(self.height))
        return (float(world_x), float(world_y))

    def _world_to_buffer_float_position(self, position: tuple[float, float]) -> tuple[float, float]:
        buffer_x = (
            float(position[0]) - float(self.paging.origin_x) + float(self.paging.buffer_origin_x)
        ) % float(self.width)
        buffer_y = (
            float(position[1]) - float(self.paging.origin_y) + float(self.paging.buffer_origin_y)
        ) % float(self.height)
        return (float(buffer_x), float(buffer_y))

    def _force_source_world_position(self, force_source: ForceSource) -> tuple[float, float]:
        if force_source.world_x is not None and force_source.world_y is not None:
            return (float(force_source.world_x), float(force_source.world_y))
        return self._buffer_to_world_float_position((float(force_source.x), float(force_source.y)))

    def _force_source_buffer_position(self, force_source: ForceSource) -> tuple[float, float]:
        if force_source.world_x is not None and force_source.world_y is not None:
            return self._world_to_buffer_float_position((float(force_source.world_x), float(force_source.world_y)))
        return (float(force_source.x), float(force_source.y))

    def _normalize_runtime_force_source(self, force_source: ForceSource) -> ForceSource:
        world_x, world_y = self._force_source_world_position(force_source)
        buffer_x, buffer_y = self._world_to_buffer_float_position((world_x, world_y))
        return replace(
            force_source,
            x=float(buffer_x),
            y=float(buffer_y),
            world_x=float(world_x),
            world_y=float(world_y),
        )

    def _buffer_gas_to_world_position(self, position: tuple[int, int]) -> tuple[int, int]:
        cell_x = int(position[0]) * int(self.gas_cell_size)
        cell_y = int(position[1]) * int(self.gas_cell_size)
        world_x, world_y = self._buffer_to_world_position((cell_x, cell_y))
        return (int(world_x // self.gas_cell_size), int(world_y // self.gas_cell_size))

    def _buffer_bbox_to_world_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = (int(value) for value in bbox)
        world_x0, world_y0 = self._buffer_to_world_position((x0, y0))
        width = max(0, x1 - x0)
        height = max(0, y1 - y0)
        return (int(world_x0), int(world_y0), int(world_x0) + width, int(world_y0) + height)

    def _clamped_world_window(self, world_x: int, world_y: int, width: int, height: int) -> tuple[int, int, int, int]:
        min_world_x = int(self.paging.origin_x)
        min_world_y = int(self.paging.origin_y)
        max_world_x = min_world_x + self.width
        max_world_y = min_world_y + self.height
        clamped_world_x = min_world_x if self.width <= 0 else max(min_world_x, min(max_world_x - 1, int(world_x)))
        clamped_world_y = min_world_y if self.height <= 0 else max(min_world_y, min(max_world_y - 1, int(world_y)))
        span_x = max(0, int(width))
        span_y = max(0, int(height))
        return (
            int(clamped_world_x),
            int(clamped_world_y),
            int(min(max_world_x, clamped_world_x + span_x)),
            int(min(max_world_y, clamped_world_y + span_y)),
        )

    def _centered_world_window(self, center_x: int, center_y: int, width: int, height: int) -> tuple[int, int, int, int]:
        clamped_center_x, clamped_center_y = self._clamp_world_position(center_x, center_y)
        min_world_x = int(self.paging.origin_x)
        min_world_y = int(self.paging.origin_y)
        max_world_x = min_world_x + self.width
        max_world_y = min_world_y + self.height
        span_x = max(0, int(width))
        span_y = max(0, int(height))
        world_x0 = max(min_world_x, int(clamped_center_x) - span_x // 2)
        world_y0 = max(min_world_y, int(clamped_center_y) - span_y // 2)
        return (
            int(world_x0),
            int(world_y0),
            int(min(max_world_x, world_x0 + span_x)),
            int(min(max_world_y, world_y0 + span_y)),
        )

    def _world_axis_spans(
        self,
        world_start: int,
        world_end: int,
        *,
        axis: str,
        gas_grid: bool = False,
    ) -> list[tuple[int, int]]:
        span = max(0, int(world_end) - int(world_start))
        if span <= 0:
            return []
        if not gas_grid:
            if axis == "x":
                size = self.width
                origin = int(self.paging.origin_x)
                buffer_origin = int(self.paging.buffer_origin_x)
            else:
                size = self.height
                origin = int(self.paging.origin_y)
                buffer_origin = int(self.paging.buffer_origin_y)
        else:
            if axis == "x":
                size = self.gas_width
                origin = int(self.paging.origin_x) // int(self.gas_cell_size)
                buffer_origin = int(self.paging.buffer_origin_x) // int(self.gas_cell_size)
            else:
                size = self.gas_height
                origin = int(self.paging.origin_y) // int(self.gas_cell_size)
                buffer_origin = int(self.paging.buffer_origin_y) // int(self.gas_cell_size)
        start = (int(world_start) - origin + buffer_origin) % size
        if span >= size:
            return [(0, size)]
        end = (start + span) % size
        if start < end:
            return [(int(start), int(end))]
        spans = [(int(start), int(size))]
        if end > 0:
            spans.append((0, int(end)))
        return spans

    def _world_axis_indices(self, world_start: int, world_end: int, *, axis: str, gas_grid: bool = False) -> np.ndarray:
        span = max(0, int(world_end) - int(world_start))
        if span <= 0:
            return np.empty((0,), dtype=np.intp)
        if not gas_grid:
            if axis == "x":
                size = self.width
                origin = int(self.paging.origin_x)
                buffer_origin = int(self.paging.buffer_origin_x)
            else:
                size = self.height
                origin = int(self.paging.origin_y)
                buffer_origin = int(self.paging.buffer_origin_y)
        else:
            if axis == "x":
                size = self.gas_width
                origin = int(self.paging.origin_x) // int(self.gas_cell_size)
                buffer_origin = int(self.paging.buffer_origin_x) // int(self.gas_cell_size)
            else:
                size = self.gas_height
                origin = int(self.paging.origin_y) // int(self.gas_cell_size)
                buffer_origin = int(self.paging.buffer_origin_y) // int(self.gas_cell_size)
        coords = np.arange(int(world_start), int(world_end), dtype=np.int64)
        return ((coords - origin + buffer_origin) % size).astype(np.intp, copy=False)

    def _extract_world_window(
        self,
        array: np.ndarray,
        world_x0: int,
        world_y0: int,
        world_x1: int,
        world_y1: int,
        *,
        x_axis: int,
        y_axis: int,
        gas_grid: bool = False,
    ) -> np.ndarray:
        x_indices = self._world_axis_indices(world_x0, world_x1, axis="x", gas_grid=gas_grid)
        y_indices = self._world_axis_indices(world_y0, world_y1, axis="y", gas_grid=gas_grid)
        window = np.take(array, y_indices, axis=y_axis)
        window = np.take(window, x_indices, axis=x_axis)
        return np.ascontiguousarray(window)

    def _world_gas_window_for_cell_world_rect(
        self,
        world_x0: int,
        world_y0: int,
        world_x1: int,
        world_y1: int,
    ) -> tuple[int, int, int, int]:
        if world_x1 <= world_x0 or world_y1 <= world_y0:
            gas_world_x0 = int(world_x0) // int(self.gas_cell_size)
            gas_world_y0 = int(world_y0) // int(self.gas_cell_size)
            return (gas_world_x0, gas_world_y0, gas_world_x0, gas_world_y0)
        return (
            int(world_x0) // int(self.gas_cell_size),
            int(world_y0) // int(self.gas_cell_size),
            int((int(world_x1) + int(self.gas_cell_size) - 1) // int(self.gas_cell_size)),
            int((int(world_y1) + int(self.gas_cell_size) - 1) // int(self.gas_cell_size)),
        )

    def _pack_cell_core_world_window(self, world_x0: int, world_y0: int, world_x1: int, world_y1: int) -> np.ndarray:
        material_id = self._extract_world_window(
            self.material_id,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        phase = self._extract_world_window(self.phase, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        cell_flags = self._extract_world_window(self.cell_flags, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        velocity = self._extract_world_window(self.velocity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        cell_temperature = self._extract_world_window(
            self.cell_temperature,
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            x_axis=1,
            y_axis=0,
        )
        timer_pack = self._extract_world_window(self.timer_pack, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)
        integrity = self._extract_world_window(self.integrity, world_x0, world_y0, world_x1, world_y1, x_axis=1, y_axis=0)

        packed = np.zeros((max(0, world_y1 - world_y0), max(0, world_x1 - world_x0), 5), dtype=np.uint32)
        packed[..., 0] = (
            material_id.astype(np.uint32)
            | (phase.astype(np.uint32) << 16)
            | (cell_flags.astype(np.uint32) << 24)
        )
        half = velocity.astype(np.float16)
        raw_half = half.view(np.uint16)
        packed[..., 1] = raw_half[..., 0].astype(np.uint32) | (raw_half[..., 1].astype(np.uint32) << 16)
        packed[..., 2] = cell_temperature.astype(np.float32).view(np.uint32)
        packed[..., 3] = (
            timer_pack[..., 0].astype(np.uint32)
            | (timer_pack[..., 1].astype(np.uint32) << 8)
            | (timer_pack[..., 2].astype(np.uint32) << 16)
            | (timer_pack[..., 3].astype(np.uint32) << 24)
        )
        packed[..., 4] = np.clip(np.rint(integrity), 0, 65535).astype(np.uint32)
        return packed

    def _world_to_buffer_clamped(self, world_x: int, world_y: int) -> tuple[int, int]:
        clamped_world_x, clamped_world_y = self._clamp_world_position(world_x, world_y)
        buffer_x, buffer_y = self.paging.world_to_buffer(clamped_world_x, clamped_world_y)
        return (int(buffer_x), int(buffer_y))

    def _clamp_world_position(self, world_x: int, world_y: int) -> tuple[int, int]:
        min_world_x = int(self.paging.origin_x)
        min_world_y = int(self.paging.origin_y)
        max_world_x = min_world_x + self.width - 1
        max_world_y = min_world_y + self.height - 1
        return (
            max(min_world_x, min(max_world_x, int(world_x))),
            max(min_world_y, min(max_world_y, int(world_y))),
        )

    def _find_nearest_empty_world_position(
        self,
        start_world_position: tuple[int, int],
        *,
        radius: int,
    ) -> tuple[int, int] | None:
        start_world_position = self._clamp_world_position(*start_world_position)
        if self._world_cell_is_empty(*start_world_position):
            return start_world_position
        if radius <= 0:
            return None
        for step in range(1, radius + 1):
            seen: set[tuple[int, int]] = set()
            for dy in range(-step, step + 1):
                for dx in range(-step, step + 1):
                    if max(abs(dx), abs(dy)) != step:
                        continue
                    world_position = self._clamp_world_position(
                        start_world_position[0] + dx,
                        start_world_position[1] + dy,
                    )
                    if world_position in seen:
                        continue
                    seen.add(world_position)
                    if self._world_cell_is_empty(*world_position):
                        return world_position
        return None

    def _world_cell_is_empty(self, world_x: int, world_y: int) -> bool:
        buffer_x, buffer_y = self._world_to_buffer_clamped(world_x, world_y)
        material_id, _ = self._material_state_for_position(
            buffer_x,
            buffer_y,
            blocked_cells=self._resolver_blocked_cells,
            released_cells=self._resolver_released_cells,
        )
        return material_id == 0

    @staticmethod
    def _world_distance_sq(left: tuple[int, int], right: tuple[int, int]) -> float:
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        return dx * dx + dy * dy

    def _entity_placeholder_bbox(self, entity_id: int) -> tuple[int, int, int, int] | None:
        cells = self.entity_placeholders.get(entity_id)
        if not cells:
            return None
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

    def _collect_observations(self, results: list[ReadbackResult]) -> dict[int, ObservationResult]:
        observations: dict[int, ObservationResult] = {}
        for result in results:
            observer_id = result.request.observer_id
            if observer_id is None:
                continue
            observations[observer_id] = ObservationResult(
                observer_id=observer_id,
                frame_id=result.frame_id,
                request=result.request,
                payload=result.payload,
            )
        return observations

    def _collect_entity_feedback(self, results: list[ReadbackResult]) -> dict[int, EntityFeedback]:
        feedback: dict[int, EntityFeedback] = {}
        for result in results:
            observer_id = result.request.observer_id
            if observer_id is None or observer_id in feedback:
                continue
            entity = self.entity_states.get(observer_id)
            if entity is None:
                continue
            entity_feedback = self._build_entity_feedback(result, entity)
            if entity_feedback is not None:
                feedback[observer_id] = entity_feedback
        return feedback

    def _build_entity_feedback(
        self,
        result: ReadbackResult,
        entity: EntityState,
    ) -> EntityFeedback | None:
        cell_payload = result.payload.get("cell")
        if cell_payload is None:
            return None
        core_words = cell_payload.get("core_words")
        entity_ids = cell_payload.get("entity_id")
        if core_words is None or entity_ids is None:
            return None

        origin_x, origin_y = (int(value) for value in cell_payload["origin"])
        width, height = (int(value) for value in cell_payload["size"])

        unpacked = unpack_cell_core(core_words)
        cells: list[EntityCellFeedback] = []
        base_world_x = int(entity.world_x) if entity.world_x is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[0])
        base_world_y = int(entity.world_y) if entity.world_y is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[1])
        for local_y in range(max(0, int(entity.height))):
            for local_x in range(max(0, int(entity.width))):
                world_x = base_world_x + local_x
                world_y = base_world_y + local_y
                lx = int(world_x) - origin_x
                ly = int(world_y) - origin_y
                if lx < 0 or ly < 0 or lx >= width or ly >= height:
                    continue
                material_id = int(unpacked["material_id"][ly, lx])
                phase = int(unpacked["phase"][ly, lx])
                integrity = float(unpacked["integrity"][ly, lx])
                occupant_entity_id = int(entity_ids[ly, lx])
                present = (
                    occupant_entity_id == entity.entity_id
                    and material_id > 0
                    and self._shadow_material_is_placeholder(material_id)
                )
                cells.append(
                    EntityCellFeedback(
                        x=int(world_x),
                        y=int(world_y),
                        present=present,
                        material_id=material_id,
                        phase=phase,
                        integrity=integrity,
                        entity_id=occupant_entity_id,
                    )
                )
        if not cells:
            return None
        bbox_xs = [cell.x for cell in cells]
        bbox_ys = [cell.y for cell in cells]
        return EntityFeedback(
            entity_id=entity.entity_id,
            bbox=(min(bbox_xs), min(bbox_ys), max(bbox_xs) + 1, max(bbox_ys) + 1),
            cells=cells,
        )

    def _build_entity_feedback_from_world(self, entity: EntityState) -> EntityFeedback | None:
        cell_state = {
            "material_id": self.material_id,
            "phase": self.phase,
            "integrity": self.integrity,
        }
        entity_runtime = {
            "entity_id": self.entity_id,
            "placeholder_displaced_material": self.placeholder_displaced_material,
        }
        return self._build_entity_feedback_from_state(entity, cell_state=cell_state, entity_runtime=entity_runtime)

    def _build_entity_feedback_from_current_state(
        self,
        entity: EntityState,
        *,
        allow_gpu_sync_readback: bool = False,
    ) -> EntityFeedback | None:
        return self._build_entity_feedback_from_state(
            entity,
            cell_state=self._current_cell_state_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback),
            entity_runtime=self._current_entity_runtime_snapshot(allow_gpu_sync_readback=allow_gpu_sync_readback),
        )

    def _build_entity_feedback_from_state(
        self,
        entity: EntityState,
        *,
        cell_state: dict[str, np.ndarray],
        entity_runtime: dict[str, np.ndarray],
    ) -> EntityFeedback | None:
        cells: list[EntityCellFeedback] = []
        base_world_x = int(entity.world_x) if entity.world_x is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[0])
        base_world_y = int(entity.world_y) if entity.world_y is not None else int(self._buffer_to_world_position((int(entity.x), int(entity.y)))[1])
        material_grid = cell_state["material_id"]
        phase_grid = cell_state["phase"]
        integrity_grid = cell_state["integrity"]
        entity_id_grid = entity_runtime["entity_id"]
        for local_y in range(max(0, int(entity.height))):
            for local_x in range(max(0, int(entity.width))):
                world_x = base_world_x + local_x
                world_y = base_world_y + local_y
                buffer_x, buffer_y = self._world_to_buffer_clamped(world_x, world_y)
                material_id = int(material_grid[buffer_y, buffer_x])
                phase = int(phase_grid[buffer_y, buffer_x])
                integrity = float(integrity_grid[buffer_y, buffer_x])
                occupant_entity_id = int(entity_id_grid[buffer_y, buffer_x])
                present = (
                    occupant_entity_id == entity.entity_id
                    and material_id > 0
                    and self._shadow_material_is_placeholder(material_id)
                )
                cells.append(
                    EntityCellFeedback(
                        x=int(world_x),
                        y=int(world_y),
                        present=present,
                        material_id=material_id,
                        phase=phase,
                        integrity=integrity,
                        entity_id=occupant_entity_id,
                    )
                )
        if not cells:
            return None
        bbox_xs = [cell.x for cell in cells]
        bbox_ys = [cell.y for cell in cells]
        return EntityFeedback(
            entity_id=int(entity.entity_id),
            bbox=(min(bbox_xs), min(bbox_ys), max(bbox_xs) + 1, max(bbox_ys) + 1),
            cells=cells,
        )

    def _capture_stripe_array(
        self,
        array: np.ndarray,
        update: PageStripeUpdate,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        spans = ranges if ranges is not None else self._stripe_buffer_ranges(update, gas_grid=False)
        parts: list[np.ndarray] = []
        for start, end in spans:
            slices = [slice(None)] * array.ndim
            slices[stripe_axis] = slice(start, end)
            parts.append(np.ascontiguousarray(array[tuple(slices)]))
        if not parts:
            return np.empty((0,), dtype=array.dtype)
        if len(parts) == 1:
            return parts[0].copy()
        return np.concatenate(parts, axis=stripe_axis)

    def _write_stripe_array(
        self,
        array: np.ndarray,
        update: PageStripeUpdate,
        values: np.ndarray,
        *,
        stripe_axis: int,
        ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        spans = ranges if ranges is not None else self._stripe_buffer_ranges(update, gas_grid=False)
        offset = 0
        for start, end in spans:
            span = end - start
            slices = [slice(None)] * array.ndim
            slices[stripe_axis] = slice(start, end)
            source_slices = [slice(None)] * values.ndim
            source_slices[stripe_axis] = slice(offset, offset + span)
            array[tuple(slices)] = values[tuple(source_slices)]
            offset += span

    def _default_page_stripe_payload(self, update: PageStripeUpdate) -> dict[str, Any]:
        cell_span = self.paging.stripe_span(update)
        gas_span = cell_span // self.gas_cell_size
        cell_width = cell_span if update.axis == "x" else self.width
        cell_height = self.height if update.axis == "x" else cell_span
        gas_width = gas_span if update.axis == "x" else self.gas_width
        gas_height = self.gas_height if update.axis == "x" else gas_span
        light_count = self.cell_optical_dose.shape[0]
        gas_count = self.gas_concentration.shape[0]
        payload = {
            "meta": {
                "axis": update.axis,
                "world_start": update.world_start,
                "world_end": update.world_end,
                "buffer_start": update.buffer_start,
                "buffer_end": update.buffer_end,
                "kind": "generated",
            },
            "cell": {
                "material_id": np.zeros((cell_height, cell_width), dtype=np.int32),
                "phase": np.zeros((cell_height, cell_width), dtype=np.uint8),
                "cell_flags": np.zeros((cell_height, cell_width), dtype=np.uint8),
                "velocity": np.zeros((cell_height, cell_width, 2), dtype=np.float32),
                "cell_temperature": np.full((cell_height, cell_width), 20.0, dtype=np.float32),
                "timer_pack": np.zeros((cell_height, cell_width, 4), dtype=np.uint8),
                "integrity": np.zeros((cell_height, cell_width), dtype=np.float32),
                "island_id": np.zeros((cell_height, cell_width), dtype=np.int32),
                "entity_id": np.zeros((cell_height, cell_width), dtype=np.int32),
                "placeholder_displaced_material": np.zeros((cell_height, cell_width), dtype=np.int32),
                "collapse_delay_pending": np.zeros((cell_height, cell_width), dtype=np.uint8),
                "visible_illumination": np.zeros((cell_height, cell_width, 3), dtype=np.float32),
                "cell_optical_dose": np.zeros((light_count, cell_height, cell_width), dtype=np.float32),
            },
            "runtime": {
                "island_ids": np.zeros((0,), dtype=np.int32),
                "island_velocity": np.zeros((0, 2), dtype=np.float32),
                "island_subcell_offset": np.zeros((0, 2), dtype=np.float32),
                "entity_placeholder_entity_id": np.zeros((cell_height, cell_width), dtype=np.int32),
            },
            "gas": {
                "ambient_temperature": np.full((gas_height, gas_width), 20.0, dtype=np.float32),
                "flow_velocity": np.zeros((gas_height, gas_width, 2), dtype=np.float32),
                "pressure_ping": np.zeros((gas_height, gas_width), dtype=np.float32),
                "gas_concentration": np.zeros((gas_count, gas_height, gas_width), dtype=np.float32),
                "gas_optical_dose": np.zeros((light_count, gas_height, gas_width), dtype=np.float32),
            },
        }
        if 0 <= self.air_gas_species_id < gas_count:
            payload["gas"]["gas_concentration"][self.air_gas_species_id] = 1.0
        return payload

    def _stripe_buffer_ranges(self, update: PageStripeUpdate, *, gas_grid: bool) -> list[tuple[int, int]]:
        if not gas_grid:
            return self.paging.stripe_buffer_ranges(update)
        cell_span = self.paging.stripe_span(update)
        if cell_span % self.gas_cell_size != 0 or update.buffer_start % self.gas_cell_size != 0:
            raise ValueError("page stripe is not aligned to the gas grid")
        size = self.gas_width if update.axis == "x" else self.gas_height
        span = min(size, cell_span // self.gas_cell_size)
        if span <= 0:
            return []
        start = (update.buffer_start // self.gas_cell_size) % size
        if span >= size:
            return [(0, size)]
        end = (start + span) % size
        if start < end:
            return [(start, end)]
        ranges = [(start, size)]
        if end > 0:
            ranges.append((0, end))
        return ranges

    def _mark_loaded_page_stripe_active(self, update: PageStripeUpdate) -> None:
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            if update.axis == "x":
                self._mark_active_rect_runtime(start, 0, end, self.height, tile_padding=1)
            else:
                self._mark_active_rect_runtime(0, start, self.width, end, tile_padding=1)

    def _rebuild_sparse_runtime_indexes(self) -> None:
        self._rebuild_entity_placeholder_index()
        self._rebuild_island_records()

    def _rebuild_entity_placeholder_index(self) -> None:
        self.entity_placeholders.clear()
        placeholder_mask = (self.entity_id > 0) & self._material_placeholder_mask(self.material_id)
        ys, xs = np.nonzero(placeholder_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            entity_id = int(self.entity_id[y, x])
            self.entity_placeholders.setdefault(entity_id, set()).add((int(x), int(y)))

    def _normalize_cell_runtime_arrays(
        self,
        material_id: np.ndarray,
        phase: np.ndarray,
        island_id: np.ndarray,
        entity_id: np.ndarray,
        placeholder_displaced_material: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phase = np.asarray(phase, dtype=np.uint8).copy()
        island_id = np.asarray(island_id, dtype=np.int32).copy()
        entity_id = np.asarray(entity_id, dtype=np.int32).copy()
        placeholder_displaced_material = np.asarray(placeholder_displaced_material, dtype=np.int32).copy()
        empty_mask = material_id <= 0
        phase[empty_mask] = 0
        placeholder_mask = self._material_placeholder_mask(material_id)
        non_placeholder_mask = empty_mask | ~placeholder_mask
        entity_id[non_placeholder_mask] = 0
        placeholder_displaced_material[non_placeholder_mask] = 0
        invalid_island_mask = (island_id > 0) & (
            (phase != int(Phase.FALLING_ISLAND)) | (material_id <= 0)
        )
        island_id[invalid_island_mask] = 0
        return phase, island_id, entity_id, placeholder_displaced_material

    def _normalize_page_stripe_cell_runtime(self, update: PageStripeUpdate) -> None:
        if self._gpu_pipeline_available(
            self.page_stripe_pipeline,
            "page stripe normalization",
            require=self.simulation_backend == "gpu",
        ):
            self.page_stripe_pipeline.normalize_cell_runtime(self, update)
            return
        self._require_cpu_oracle_backend("page stripe normalization")
        cell_axis = 1 if update.axis == "x" else 0
        material_id = self._capture_stripe_array(self.material_id, update, stripe_axis=cell_axis)
        phase = self._capture_stripe_array(self.phase, update, stripe_axis=cell_axis)
        island_id = self._capture_stripe_array(self.island_id, update, stripe_axis=cell_axis)
        entity_id = self._capture_stripe_array(self.entity_id, update, stripe_axis=cell_axis)
        placeholder_displaced_material = self._capture_stripe_array(
            self.placeholder_displaced_material,
            update,
            stripe_axis=cell_axis,
        )
        phase, island_id, entity_id, placeholder_displaced_material = self._normalize_cell_runtime_arrays(
            material_id,
            phase,
            island_id,
            entity_id,
            placeholder_displaced_material,
        )
        self._write_stripe_array(self.phase, update, phase, stripe_axis=cell_axis)
        self._write_stripe_array(self.island_id, update, island_id, stripe_axis=cell_axis)
        self._write_stripe_array(self.entity_id, update, entity_id, stripe_axis=cell_axis)
        self._write_stripe_array(
            self.placeholder_displaced_material,
            update,
            placeholder_displaced_material,
            stripe_axis=cell_axis,
        )

    def _capture_page_stripe_entity_placeholder_runtime(
        self,
        update: PageStripeUpdate,
        *,
        stripe_axis: int,
    ) -> np.ndarray:
        live_placeholder_mask = (self.entity_id > 0) & self._material_placeholder_mask(self.material_id)
        grid = np.where(live_placeholder_mask, self.entity_id, 0).astype(np.int32)
        for entity_id, cells in self.entity_placeholders.items():
            for x, y in cells:
                if self.in_bounds(x, y):
                    grid[y, x] = int(entity_id)
        return self._capture_stripe_array(grid, update, stripe_axis=stripe_axis)

    def _apply_page_stripe_entity_placeholder_runtime(
        self,
        update: PageStripeUpdate,
        entity_placeholder_entity_id: np.ndarray | None,
    ) -> None:
        cell_axis = 1 if update.axis == "x" else 0
        if entity_placeholder_entity_id is None:
            if self.simulation_backend == "gpu":
                raise RuntimeError(
                    "GPU page stripe placeholder runtime requires payload runtime data; CPU fallback is disabled"
                )
            placeholder_mask = (self.entity_id > 0) & self._material_placeholder_mask(self.material_id)
            dense_placeholder_ids = np.where(placeholder_mask, self.entity_id, 0).astype(np.int32)
            entity_placeholder_entity_id = self._capture_stripe_array(
                dense_placeholder_ids,
                update,
                stripe_axis=cell_axis,
            )
        spans = self._stripe_buffer_ranges(update, gas_grid=False)
        if update.axis == "x":
            def cell_in_loaded_stripe(cell: tuple[int, int]) -> bool:
                return any(start <= cell[0] < end for start, end in spans)
        else:
            def cell_in_loaded_stripe(cell: tuple[int, int]) -> bool:
                return any(start <= cell[1] < end for start, end in spans)

        next_entity_cells: dict[int, set[tuple[int, int]]] = {}
        for entity_id, cells in self.entity_placeholders.items():
            filtered = {cell for cell in cells if not cell_in_loaded_stripe(cell)}
            if filtered:
                next_entity_cells[int(entity_id)] = filtered

        offset = 0
        for start, end in spans:
            span = end - start
            if update.axis == "x":
                stripe_slice = entity_placeholder_entity_id[:, offset : offset + span]
            else:
                stripe_slice = entity_placeholder_entity_id[offset : offset + span, :]
            ys, xs = np.nonzero(stripe_slice > 0)
            for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                entity_id = int(stripe_slice[local_y, local_x])
                if entity_id <= 0:
                    continue
                cell = (start + local_x, local_y) if update.axis == "x" else (local_x, start + local_y)
                next_entity_cells.setdefault(entity_id, set()).add(cell)
            offset += span
        self.entity_placeholders = next_entity_cells

    def _rebuild_island_records(self) -> None:
        previous_records = dict(self.islands)
        self.islands.clear()
        invalid_island_mask = (self.island_id > 0) & (
            (self.phase != int(Phase.FALLING_ISLAND)) | (self.material_id <= 0)
        )
        if np.any(invalid_island_mask):
            self.island_id[invalid_island_mask] = 0
        for island_id in np.unique(self.island_id):
            island_id = int(island_id)
            if island_id <= 0:
                continue
            coords = np.argwhere(
                (self.island_id == island_id)
                & (self.phase == int(Phase.FALLING_ISLAND))
                & (self.material_id > 0)
            )
            if coords.size == 0:
                continue
            min_y, min_x = coords.min(axis=0).tolist()
            max_y, max_x = coords.max(axis=0).tolist()
            previous = previous_records.get(island_id)
            if previous is None:
                velocity_xy = tuple(np.mean(self.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32).tolist())
                subcell_offset = (0.0, 0.0)
            else:
                velocity_xy = (float(previous.velocity_xy[0]), float(previous.velocity_xy[1]))
                subcell_offset = (float(previous.subcell_offset[0]), float(previous.subcell_offset[1]))
            self.islands[island_id] = FallingIslandRecord(
                island_id=island_id,
                bbox=(int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1),
                velocity_xy=(float(velocity_xy[0]), float(velocity_xy[1])),
                subcell_offset=subcell_offset,
            )
        self.next_island_id = max(1, max(self.islands, default=0) + 1)

    def _capture_page_stripe_island_runtime(self, stripe_island_ids: np.ndarray) -> dict[str, np.ndarray]:
        island_ids = sorted(int(island_id) for island_id in np.unique(stripe_island_ids) if int(island_id) > 0)
        if not island_ids:
            return {
                "island_ids": np.zeros((0,), dtype=np.int32),
                "island_velocity": np.zeros((0, 2), dtype=np.float32),
                "island_subcell_offset": np.zeros((0, 2), dtype=np.float32),
            }
        velocity = np.zeros((len(island_ids), 2), dtype=np.float32)
        subcell_offset = np.zeros((len(island_ids), 2), dtype=np.float32)
        for index, island_id in enumerate(island_ids):
            record = self.islands.get(island_id)
            if record is None:
                coords = np.argwhere(
                    (self.island_id == island_id)
                    & (self.phase == int(Phase.FALLING_ISLAND))
                    & (self.material_id > 0)
                )
                if coords.size != 0:
                    mean_velocity = np.mean(self.velocity[coords[:, 0], coords[:, 1]], axis=0).astype(np.float32)
                    velocity[index] = mean_velocity
                continue
            velocity[index] = np.asarray(record.velocity_xy, dtype=np.float32)
            subcell_offset[index] = np.asarray(record.subcell_offset, dtype=np.float32)
        return {
            "island_ids": np.asarray(island_ids, dtype=np.int32),
            "island_velocity": velocity,
            "island_subcell_offset": subcell_offset,
        }

    def _page_stripe_island_bboxes_from_payload(
        self,
        update: PageStripeUpdate,
        payload: dict[str, Any],
    ) -> dict[int, tuple[int, int, int, int]] | None:
        cell_payload = payload.get("cell", {})
        try:
            material_id = np.asarray(cell_payload["material_id"], dtype=np.int32)
            phase = np.asarray(cell_payload["phase"], dtype=np.uint8)
            island_id = np.asarray(cell_payload["island_id"], dtype=np.int32)
        except KeyError:
            return None
        if material_id.shape != phase.shape or material_id.shape != island_id.shape:
            return None
        valid = (island_id > 0) & (material_id > 0) & (phase == int(Phase.FALLING_ISLAND))
        if not np.any(valid):
            return {}
        boxes: dict[int, list[int]] = {}
        offset = 0
        for start, end in self._stripe_buffer_ranges(update, gas_grid=False):
            span = int(end) - int(start)
            if span <= 0:
                continue
            if update.axis == "x":
                stripe = valid[:, offset : offset + span]
                stripe_ids = island_id[:, offset : offset + span]
                ys, xs = np.nonzero(stripe)
                for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                    current_id = int(stripe_ids[local_y, local_x])
                    x = int(start) + int(local_x)
                    y = int(local_y)
                    box = boxes.setdefault(current_id, [x, y, x + 1, y + 1])
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x + 1)
                    box[3] = max(box[3], y + 1)
            else:
                stripe = valid[offset : offset + span, :]
                stripe_ids = island_id[offset : offset + span, :]
                ys, xs = np.nonzero(stripe)
                for local_y, local_x in zip(ys.tolist(), xs.tolist()):
                    current_id = int(stripe_ids[local_y, local_x])
                    x = int(local_x)
                    y = int(start) + int(local_y)
                    box = boxes.setdefault(current_id, [x, y, x + 1, y + 1])
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x + 1)
                    box[3] = max(box[3], y + 1)
            offset += span
        return {island_id: tuple(box) for island_id, box in boxes.items()}

    def _merge_island_runtime_payload(
        self,
        runtime_payload: dict[str, Any] | None,
        *,
        update: PageStripeUpdate | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not runtime_payload:
            return
        island_ids = np.asarray(runtime_payload.get("island_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        island_velocity = np.asarray(runtime_payload.get("island_velocity", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        island_subcell_offset = np.asarray(runtime_payload.get("island_subcell_offset", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        payload_bboxes = (
            self._page_stripe_island_bboxes_from_payload(update, payload)
            if update is not None and payload is not None
            else None
        )
        count = min(len(island_ids), len(island_velocity), len(island_subcell_offset))
        for index in range(count):
            island_id = int(island_ids[index])
            if island_id <= 0:
                continue
            if payload_bboxes is not None and island_id not in payload_bboxes:
                self.islands.pop(island_id, None)
                continue
            previous = self.islands.get(island_id)
            bbox = (
                payload_bboxes[island_id]
                if payload_bboxes is not None
                else (0, 0, 0, 0) if previous is None else previous.bbox
            )
            self.islands[island_id] = FallingIslandRecord(
                island_id=island_id,
                bbox=bbox,
                velocity_xy=(float(island_velocity[index, 0]), float(island_velocity[index, 1])),
                subcell_offset=(float(island_subcell_offset[index, 0]), float(island_subcell_offset[index, 1])),
            )

    def _rebuild_material_property_arrays(self) -> None:
        max_id = max(self.rulebook.materials_by_id, default=0)
        size = max_id + 1
        self.material_base_color = np.zeros((size, 3), dtype=np.float32)
        self.material_density = np.zeros(size, dtype=np.float32)
        self.material_gravity = np.zeros(size, dtype=np.float32)
        self.material_wind = np.zeros(size, dtype=np.float32)
        self.material_drag = np.zeros(size, dtype=np.float32)
        self.material_friction = np.zeros(size, dtype=np.float32)
        self.material_elasticity = np.zeros(size, dtype=np.float32)
        self.material_max_dda_step = np.zeros(size, dtype=np.int32)
        self.material_default_phase = np.zeros(size, dtype=np.uint8)
        self.material_base_integrity = np.zeros(size, dtype=np.float32)
        self.material_spawn_temperature = np.full(size, np.nan, dtype=np.float32)
        self.material_reaction_slots = np.full((size, 8), -1, dtype=np.int32)
        self.material_material_tag_mask = np.zeros(size, dtype=np.uint32)
        self.material_gas_tag_mask = np.zeros(size, dtype=np.uint32)
        self.material_light_tag_mask = np.zeros(size, dtype=np.uint32)
        self.material_powder_solver_kind = np.zeros(size, dtype=np.uint8)
        self.material_liquid_solver_kind = np.zeros(size, dtype=np.uint8)
        self.material_falling_island_break_kind = np.zeros(size, dtype=np.uint8)
        self.material_heat_capacity = np.zeros(size, dtype=np.float32)
        self.material_conductivity = np.zeros(size, dtype=np.float32)
        self.material_ambient_exchange = np.zeros(size, dtype=np.float32)
        self.material_is_structural = np.zeros(size, dtype=np.bool_)
        self.material_is_support_anchor = np.zeros(size, dtype=np.bool_)
        self.material_is_plant = np.zeros(size, dtype=np.bool_)
        self.material_is_placeholder = np.zeros(size, dtype=np.bool_)
        self.material_collapse_behavior = np.zeros(size, dtype=np.uint8)
        self.material_collapse_generation_id = np.zeros(size, dtype=np.int32)
        self.material_powder_generation_id = np.zeros(size, dtype=np.int32)
        self.material_name_by_id = [""] * size
        self.placeholder_material_id = 0
        for material_id, material in self.rulebook.materials_by_id.items():
            self.material_base_color[material_id] = np.asarray(material.base_color, dtype=np.float32)
            self.material_density[material_id] = material.density
            self.material_gravity[material_id] = material.gravity_scale
            self.material_wind[material_id] = material.wind_coupling
            self.material_drag[material_id] = material.drag_scale
            self.material_friction[material_id] = material.friction
            self.material_elasticity[material_id] = material.elasticity
            self.material_max_dda_step[material_id] = int(material.max_dda_step)
            self.material_default_phase[material_id] = np.uint8(int(material.default_phase))
            self.material_base_integrity[material_id] = material.base_integrity
            self.material_spawn_temperature[material_id] = (
                np.float32(material.spawn_temperature) if material.spawn_temperature is not None else np.float32(np.nan)
            )
            self.material_reaction_slots[material_id] = np.asarray(material.reaction_slots, dtype=np.int32)
            self.material_material_tag_mask[material_id] = np.uint32(material.material_tag_mask)
            self.material_gas_tag_mask[material_id] = np.uint32(material.gas_tag_mask)
            self.material_light_tag_mask[material_id] = np.uint32(material.light_tag_mask)
            self.material_powder_solver_kind[material_id] = np.uint8(POWDER_SOLVER_KIND_IDS.get(material.powder_solver_kind, 0))
            self.material_liquid_solver_kind[material_id] = np.uint8(LIQUID_SOLVER_KIND_IDS.get(material.liquid_solver_kind, 0))
            self.material_falling_island_break_kind[material_id] = np.uint8(
                FALLING_ISLAND_BREAK_KIND_IDS.get(material.falling_island_break_kind, 0)
            )
            self.material_heat_capacity[material_id] = material.heat_capacity
            self.material_conductivity[material_id] = material.conductivity
            self.material_ambient_exchange[material_id] = material.ambient_exchange_rate
            self.material_is_structural[material_id] = material.is_structural
            self.material_is_support_anchor[material_id] = material.is_support_anchor
            self.material_is_plant[material_id] = material.render_group == "plant" or "plant" in material.tags
            self.material_is_placeholder[material_id] = material.render_group == "placeholder" or "placeholder" in material.tags
            self.material_collapse_behavior[material_id] = np.uint8(COLLAPSE_BEHAVIOR_IDS.get(material.collapse_behavior, 0))
            self.material_collapse_generation_id[material_id] = int(
                self.rulebook.material_id(material.collapse_generation) if material.collapse_generation else 0
            )
            self.material_powder_generation_id[material_id] = int(
                self.rulebook.material_id(material.powder_generation) if material.powder_generation else 0
            )
            self.material_name_by_id[material_id] = material.name
            if material.name == "placeholder_solid":
                self.placeholder_material_id = int(material_id)
        chaos_convert_bit = int(self.tag_bits_by_name.get("chaos_convert", 0))
        self.random_convert_material_ids = [
            int(material_id)
            for material_id, material in sorted(self.rulebook.materials_by_id.items())
            if chaos_convert_bit != 0
            and bool(int(material.material_tag_mask) & chaos_convert_bit)
            and int(material.default_phase) == int(Phase.POWDER)
        ]

    def _rebuild_gas_property_arrays(self) -> None:
        max_id = max(self.rulebook.gases_by_id, default=-1)
        size = max(0, max_id + 1)
        self.gas_material_reaction_tag_mask = np.zeros(size, dtype=np.uint32)
        self.gas_light_reaction_tag_mask = np.zeros(size, dtype=np.uint32)
        self.gas_density_factor = np.zeros(size, dtype=np.float32)
        self.gas_condense_material_id = np.zeros(size, dtype=np.int32)
        self.gas_name_by_id = [""] * size
        self.air_gas_species_id = -1
        for species_id, gas in self.rulebook.gases_by_id.items():
            self.gas_material_reaction_tag_mask[species_id] = np.uint32(gas.material_reaction_tag_mask)
            self.gas_light_reaction_tag_mask[species_id] = np.uint32(gas.light_reaction_tag_mask)
            self.gas_density_factor[species_id] = np.float32(gas.density_factor)
            self.gas_condense_material_id[species_id] = int(
                self.rulebook.material_id(gas.condense_to_material) if gas.condense_to_material else 0
            )
            self.gas_name_by_id[species_id] = gas.name
            if gas.name == "air":
                self.air_gas_species_id = int(species_id)
        if self.air_gas_species_id < 0:
            gas_table = self.bridge.shadow_typed_tables.get("gas_table")
            if gas_table is not None:
                air_species_id = int(typed_gas_id(gas_table, "air"))
                if 0 <= air_species_id < size:
                    self.air_gas_species_id = air_species_id

    def _rebuild_light_property_arrays(self) -> None:
        max_id = max(self.rulebook.lights_by_id, default=-1)
        size = max(0, max_id + 1)
        self.light_default_range = np.zeros(size, dtype=np.int32)
        self.light_dose_channel = np.zeros(size, dtype=np.int32)
        self.light_color = np.zeros((size, 3), dtype=np.float32)
        self.light_name_by_id = [""] * size
        for light_id, light in self.rulebook.lights_by_id.items():
            self.light_default_range[light_id] = int(light.default_range)
            self.light_dose_channel[light_id] = int(light.dose_channel_id)
            self.light_color[light_id] = np.asarray(light.color, dtype=np.float32)
            self.light_name_by_id[light_id] = light.name

    def _cell_participates_in_collapse(self, material_id: int, phase: int) -> bool:
        return (
            material_id != 0
            and phase != int(Phase.FALLING_ISLAND)
            and (
                self.material_is_structural[material_id]
                or self.material_is_support_anchor[material_id]
            )
        )

    def _mark_collapse_dirty_rect(self, x0: int, y0: int, x1: int, y1: int, *, pad: int = 8) -> None:
        self.collapse_dirty_regions.append(
            (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(self.width, x1 + pad),
                min(self.height, y1 + pad),
            )
        )

    def _drain_gpu_collapse_structure_dirty_tiles(self) -> None:
        regions = drain_collapse_structure_dirty_tile_regions(self)
        if regions:
            self.collapse_dirty_regions.extend(regions)

    def _paint_material(self, x: int, y: int, material: str, radius: int) -> None:
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        self.set_material_by_mask(mask, material)

    def _write_material_region_immediate(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        material: str,
    ) -> None:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, int(x) + max(0, int(width)))
        y1 = min(self.height, int(y) + max(0, int(height)))
        if x0 >= x1 or y0 >= y1:
            return
        for write_y in range(y0, y1):
            for write_x in range(x0, x1):
                self.set_cell(write_x, write_y, material)

    def _build_demo_scene(self) -> None:
        active_w = int(self.paging.active_width)
        active_h = int(self.paging.active_height)
        floor_y = max(0, active_h - 28)
        self._fill_rect(0, floor_y, active_w, 28, "raw_stone_solid")
        self._fill_rect(32, floor_y - 58, 160, 14, "sandstone_solid")
        self._fill_rect(60, floor_y - 112, 118, 54, "sand_powder")
        self._fill_rect(230, floor_y - 24, 112, 24, "water_liquid")
        self._fill_rect(374, floor_y - 18, 78, 18, "oil_liquid")
        self._fill_rect(500, floor_y - 86, 12, 86, "raw_stone_solid")
        self._fill_rect(520, floor_y - 46, 130, 18, "sandstone_solid")
        self._fill_rect(550, floor_y - 86, 76, 40, "gravel_powder")
        self._fill_rect(690, floor_y - 140, 72, 16, "log_solid")
        self._fill_rect(700, floor_y - 188, 12, 48, "root_solid")

    def _fill_rect(self, x: int, y: int, width: int, height: int, material: str) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        material_id = self._resolve_sanctioned_material_id(material)
        if material_id <= 0:
            raise KeyError(material)
        phase = int(self.material_default_phase[material_id]) if material_id < self.material_default_phase.shape[0] else 0
        integrity = (
            float(self.material_base_integrity[material_id])
            if material_id < self.material_base_integrity.shape[0]
            else 0.0
        )
        self.material_id[y0:y1, x0:x1] = int(material_id)
        self.phase[y0:y1, x0:x1] = phase
        self.cell_flags[y0:y1, x0:x1] = 0
        self.velocity[y0:y1, x0:x1] = 0.0
        self.timer_pack[y0:y1, x0:x1] = 0
        self.integrity[y0:y1, x0:x1] = integrity
        self.island_id[y0:y1, x0:x1] = 0
        self.entity_id[y0:y1, x0:x1] = 0
        self.placeholder_displaced_material[y0:y1, x0:x1] = 0
        if material_id < self.material_spawn_temperature.shape[0]:
            spawn_temperature = float(self.material_spawn_temperature[material_id])
            if np.isfinite(spawn_temperature):
                self.cell_temperature[y0:y1, x0:x1] = np.maximum(
                    self.cell_temperature[y0:y1, x0:x1],
                    spawn_temperature,
                )
