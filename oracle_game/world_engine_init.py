from __future__ import annotations

from collections import deque
import threading
from typing import Any, TYPE_CHECKING

import numpy as np

from oracle_game.active import ActiveRegionTracker
from oracle_game.gpu import GPUBridge
from oracle_game.page_store import InMemoryPageStore, PageStore
from oracle_game.paging import RingPagingWindow
from oracle_game.rules import RuleBook
from oracle_game.sim.collapse import CollapseSolver
from oracle_game.sim.gas import GasSolver
from oracle_game.sim.heat import HeatSolver
from oracle_game.sim.liquid import LiquidSolver
from oracle_game.sim.motion import MotionSolver
from oracle_game.sim.optics import OpticsSolver
from oracle_game.sim.gpu_placeholders import GPUPlaceholderPipeline
from oracle_game.sim.gpu_page_stripes import GPUPageStripePipeline
from oracle_game.sim.gpu_world_commands import GPUWorldCommandPipeline
from oracle_game.sim.reactions import ReactionSolver
from oracle_game.sim.gpu_merge import GPUMergePipeline
from oracle_game.types import (
    DebugView, EntityPlaceholder, EntityState, ForceSource, PageStripeUpdate,
    ReadbackRequest, ReadbackResult, WorldCommand, WorldFrameInput, WorldFrameOutput,
)
from oracle_game.world_constants import GPU_REALTIME_BUDGET_CELL_THRESHOLD

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def _init_world_engine(
    engine: WorldEngine,
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
    engine.simulation_backend = simulation_backend
    engine._world_simulation_frame_active = False
    engine.width = width
    engine.height = height
    engine.gas_cell_size = gas_cell_size
    engine.gas_width = max(1, (width + gas_cell_size - 1) // gas_cell_size)
    engine.gas_height = max(1, (height + gas_cell_size - 1) // gas_cell_size)
    engine.paging = RingPagingWindow(width, height, active_width or width // 2, active_height or height // 2)
    engine.active = ActiveRegionTracker(width, height)
    engine.rulebook = RuleBook()
    engine.bridge = (
        GPUBridge(create_standalone=False)
        if simulation_backend == "cpu" and gpu_context is None
        else GPUBridge(ctx=gpu_context)
    )
    if simulation_backend == "gpu" and not engine._gpu_context_available():
        raise RuntimeError("GPU world simulation requires a ModernGL 4.3+ context; CPU fallback is disabled")
    engine.page_store = InMemoryPageStore() if page_store is None else page_store
    engine.frame_id = 0
    engine.state_lock = threading.RLock()
    engine.command_queue: deque[WorldCommand] = deque()
    engine.pending_frame_inputs: deque[WorldFrameInput] = deque()
    engine.completed_frame_outputs: deque[WorldFrameOutput] = deque()
    engine.canceled_frame_submission_ids: set[int] = set()
    engine.next_frame_submission_id = 1
    engine.next_readback_request_id = 1
    engine.pending_readbacks: list[ReadbackRequest] = []
    engine.inflight_readbacks: list[ReadbackRequest] = []
    engine.completed_readbacks: deque[ReadbackResult] = deque()
    engine.canceled_readback_request_ids: set[int] = set()
    engine.last_entity_observation_consume_snapshot: dict[str, Any] = {
        "frame_id": 0,
        "consumed": 0,
        "consumed_readbacks": [],
        "observations": {},
        "entity_feedback": {},
    }
    engine.controller_state_snapshot: Any = None
    engine.bootstrap_log: list[str] = []
    engine.bridge_frame_commands: list[WorldCommand] = []
    engine.bridge_frame_readback_requests: list[ReadbackRequest] = []
    engine.bridge_frame_placeholders: list[EntityPlaceholder] = []
    engine.bridge_frame_placeholder_dirty_rects: list[tuple[int, int, int, int]] = []
    engine._pending_placeholder_dirty_rects: list[tuple[int, int, int, int]] = []
    engine.bridge_frame_paging_updates: list[PageStripeUpdate] = []
    engine.bridge_frame_page_stripes: list[tuple[PageStripeUpdate, dict[str, Any]]] = []
    engine._bridge_inputs_prepared = False
    engine._gpu_cpu_dirty_resources: set[str] = set()
    engine._resolver_blocked_cells: set[tuple[int, int]] | None = None
    engine._resolver_released_cells: set[tuple[int, int]] | None = None
    engine.force_sources: list[ForceSource] = []
    engine.persistent_emitters: list[dict[str, object]] = []
    engine.emitters: list[dict[str, object]] = []
    engine._formal_gpu_frame_has_light_dose: bool | None = None
    engine._gpu_optics_outputs_clear = False
    engine.gpu_realtime_budget_enabled = True
    engine.gpu_realtime_budget_cell_threshold = GPU_REALTIME_BUDGET_CELL_THRESHOLD
    engine.profile_passes_enabled = False
    engine.profile_passes_sync = False
    engine.last_pass_profile: dict[str, Any] = {"passes": [], "summary": {}, "skipped_stages": []}
    engine.last_skipped_gpu_stages: list[str] = []
    engine.formal_collapse_interval_frames = 4
    engine.collapse_dirty_regions: list[tuple[int, int, int, int]] = []
    engine.collapse_deferred_regions: list[tuple[int, int, int, int]] = []
    engine._gpu_collapse_structure_dirty_tiles_pending = False
    engine._gpu_collapse_structure_dirty_tiles_deferred = False
    engine.islands: dict[int, object] = {}
    engine.entity_states: dict[int, EntityState] = {}
    engine.entity_placeholders: dict[int, set[tuple[int, int]]] = {}
    engine.next_island_id = 1

    engine.material_id = np.zeros((height, width), dtype=np.int32)
    engine.phase = np.zeros((height, width), dtype=np.uint8)
    engine.cell_flags = np.zeros((height, width), dtype=np.uint8)
    engine.velocity = np.zeros((height, width, 2), dtype=np.float32)
    engine.cell_temperature = np.full((height, width), 20.0, dtype=np.float32)
    engine.timer_pack = np.zeros((height, width, 4), dtype=np.uint8)
    engine.integrity = np.zeros((height, width), dtype=np.float32)
    engine.island_id = np.zeros((height, width), dtype=np.int32)
    engine.entity_id = np.zeros((height, width), dtype=np.int32)
    engine.placeholder_displaced_material = np.zeros((height, width), dtype=np.int32)
    engine.collapse_delay_pending = np.zeros((height, width), dtype=np.bool_)

    engine.flow_velocity = np.zeros((engine.gas_height, engine.gas_width, 2), dtype=np.float32)
    engine.ambient_temperature = np.full((engine.gas_height, engine.gas_width), 20.0, dtype=np.float32)
    engine.pressure_ping = np.zeros((engine.gas_height, engine.gas_width), dtype=np.float32)
    engine.gas_concentration = np.zeros((1, engine.gas_height, engine.gas_width), dtype=np.float32)
    engine.visible_illumination = np.zeros((height, width, 3), dtype=np.float32)
    engine.cell_optical_dose = np.zeros((1, height, width), dtype=np.float32)
    engine.gas_optical_dose = np.zeros((1, engine.gas_height, engine.gas_width), dtype=np.float32)
    engine.default_debug_view = DebugView.MATERIAL

    engine.material_density = np.zeros(1, dtype=np.float32)
    engine.material_base_color = np.zeros((1, 3), dtype=np.float32)
    engine.material_gravity = np.zeros(1, dtype=np.float32)
    engine.material_wind = np.zeros(1, dtype=np.float32)
    engine.material_drag = np.zeros(1, dtype=np.float32)
    engine.material_friction = np.zeros(1, dtype=np.float32)
    engine.material_elasticity = np.zeros(1, dtype=np.float32)
    engine.material_max_dda_step = np.zeros(1, dtype=np.int32)
    engine.material_default_phase = np.zeros(1, dtype=np.uint8)
    engine.material_base_integrity = np.zeros(1, dtype=np.float32)
    engine.material_spawn_temperature = np.full(1, np.nan, dtype=np.float32)
    engine.material_reaction_slots = np.full((1, 8), -1, dtype=np.int32)
    engine.material_material_tag_mask = np.zeros(1, dtype=np.uint32)
    engine.material_gas_tag_mask = np.zeros(1, dtype=np.uint32)
    engine.material_light_tag_mask = np.zeros(1, dtype=np.uint32)
    engine.material_powder_solver_kind = np.zeros(1, dtype=np.uint8)
    engine.material_liquid_solver_kind = np.zeros(1, dtype=np.uint8)
    engine.material_falling_island_break_kind = np.zeros(1, dtype=np.uint8)
    engine.material_heat_capacity = np.zeros(1, dtype=np.float32)
    engine.material_conductivity = np.zeros(1, dtype=np.float32)
    engine.material_ambient_exchange = np.zeros(1, dtype=np.float32)
    engine.material_is_structural = np.zeros(1, dtype=np.bool_)
    engine.material_is_support_anchor = np.zeros(1, dtype=np.bool_)
    engine.material_is_plant = np.zeros(1, dtype=np.bool_)
    engine.material_is_placeholder = np.zeros(1, dtype=np.bool_)
    engine.material_collapse_behavior = np.zeros(1, dtype=np.uint8)
    engine.material_collapse_generation_id = np.zeros(1, dtype=np.int32)
    engine.material_powder_generation_id = np.zeros(1, dtype=np.int32)
    engine.material_name_by_id: list[str] = [""]
    engine.tag_bits_by_name: dict[str, int] = {}
    engine.random_convert_material_ids: list[int] = []
    engine.placeholder_material_id = 0

    engine.gas_material_reaction_tag_mask = np.zeros(1, dtype=np.uint32)
    engine.gas_light_reaction_tag_mask = np.zeros(1, dtype=np.uint32)
    engine.gas_density_factor = np.zeros(1, dtype=np.float32)
    engine.gas_condense_material_id = np.zeros(1, dtype=np.int32)
    engine.gas_name_by_id: list[str] = []
    engine.air_gas_species_id = -1

    engine.light_default_range = np.zeros(1, dtype=np.int32)
    engine.light_dose_channel = np.zeros(1, dtype=np.int32)
    engine.light_color = np.zeros((1, 3), dtype=np.float32)
    engine.light_name_by_id: list[str] = [""]
    engine._stable_shadow_payloads: dict[str, Any] = {}

    engine.gas_solver = GasSolver()
    engine.heat_solver = HeatSolver()
    engine.collapse_solver = CollapseSolver()
    engine.motion_solver = MotionSolver()
    engine.liquid_solver = LiquidSolver()
    engine.optics_solver = OpticsSolver()
    engine.reaction_solver = ReactionSolver()
    engine.merge_pipeline = GPUMergePipeline()
    engine.placeholder_pipeline = GPUPlaceholderPipeline()
    engine.page_stripe_pipeline = GPUPageStripePipeline()
    engine.grid_command_pipeline = GPUWorldCommandPipeline()

    engine.bootstrap_defaults()
    engine.reset_world()
    engine.bridge.sync_world(engine)
    if engine.simulation_backend == "gpu":
        engine.bridge.mark_gpu_authoritative(
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
        engine._gpu_cpu_dirty_resources.clear()
    engine._closed = False
