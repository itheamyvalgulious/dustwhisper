from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag, auto
from typing import Any


class Phase(IntEnum):
    STATIC_SOLID = 0
    POWDER = 1
    LIQUID = 2
    FALLING_ISLAND = 3


class CollapseBehavior(IntEnum):
    NONE = 0
    FALLING_ISLAND = 1
    DELAYED = 2
    IMMUNE = 3


COLLAPSE_BEHAVIOR_IDS: dict[str, int] = {
    "falling_island": int(CollapseBehavior.FALLING_ISLAND),
    "delayed": int(CollapseBehavior.DELAYED),
    "immune": int(CollapseBehavior.IMMUNE),
}


class CellFlag(IntFlag):
    NONE = 0
    PHASE_LOCKED = 1 << 0
    REACTION_LATCHED = 1 << 1
    RECENTLY_CONVERTED = 1 << 2


class ReactionType(Enum):
    NONE = auto()
    EMIT_MATERIAL = auto()
    EMIT_LIGHT = auto()
    MODIFY_GAS = auto()
    CONVERT_MATERIAL = auto()
    MODIFY_TEMPERATURE = auto()
    HARM = auto()


class DebugView(Enum):
    MATERIAL = "material"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    HEAT = "heat"
    LIQUID = "liquid"
    REACTION = "reaction"
    COLLAPSE = "collapse"
    OPTICS = "optics"
    VELOCITY = "velocity"
    LIGHT = "light"
    GAS = "gas"
    MOTION = "motion"
    ACTIVE = "active"


class Direction(Enum):
    ALL = "all"
    RANDOM = "random"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SPEED = "speed"


@dataclass(slots=True)
class ReactionAction:
    reaction_type: ReactionType
    target_material: str | None = None
    emit_material: str | None = None
    light_type: str | None = None
    gas_species: str | None = None
    duration: int = 0
    speed: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    direction: Direction = Direction.ALL
    strength: float = 0.0
    beam_width: float = 1.0
    range_cells: int = 0
    delta: float = 0.0
    value: float = 0.0
    generation: int = 0
    harm_per_frame: float = 0.0
    integrity_threshold: float = 0.0
    allow_subunit_scale: bool = False


@dataclass(slots=True)
class MaterialDef:
    material_id: int
    name: str
    display_name: str
    default_phase: Phase
    render_group: str
    base_color: tuple[float, float, float]
    density: float
    gravity_scale: float
    wind_coupling: float
    drag_scale: float
    friction: float
    elasticity: float
    max_dda_step: int
    powder_solver_kind: str
    liquid_solver_kind: str
    falling_island_break_kind: str
    is_structural: bool
    is_support_anchor: bool
    collapse_behavior: str
    collapse_generation: str | None
    powder_generation: str | None
    base_integrity: float
    heat_capacity: float
    conductivity: float
    ambient_exchange_rate: float
    melt_point: float | None
    boil_point: float | None
    melt_to_material: str | None
    freeze_to_material: str | None
    boil_to_gas_species: str | None
    spawn_temperature: float | None = None
    material_tag_mask: int = 0
    gas_tag_mask: int = 0
    light_tag_mask: int = 0
    reaction_slots: tuple[int, ...] = (-1, -1, -1, -1, -1, -1, -1, -1)
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class GasSpeciesDef:
    species_id: int
    name: str
    display_name: str
    color: tuple[float, float, float]
    diffusion_rate: float
    buoyancy: float
    decay_rate: float
    temperature_coupling: float
    condense_point: float | None
    condense_to_material: str | None
    pressure_factor: float
    density_factor: float
    material_reaction_tag_mask: int = 0
    light_reaction_tag_mask: int = 0


@dataclass(slots=True)
class LightTypeDef:
    light_type_id: int
    name: str
    display_name: str
    color: tuple[float, float, float]
    visual_channel: int
    default_range: int
    max_bounce: int
    dose_channel_id: int
    render_style: str


@dataclass(slots=True)
class MaterialOpticsDef:
    material_name: str
    light_type: str
    absorption: float
    scattering: float
    refraction: float


@dataclass(slots=True)
class PairReactionRule:
    lhs_material: str | None = None
    lhs_gas: str | None = None
    rhs_material: str | None = None
    rhs_gas: str | None = None
    rhs_light: str | None = None
    lhs_tag_mask: int = 0
    rhs_tag_mask: int = 0
    phases: tuple[Phase, ...] = ()
    min_temperature: float = float("-inf")
    max_temperature: float = float("inf")
    threshold: float = 0.0
    rate: float = 1.0
    consume_policy: str = "none"
    result_action: int = -1
    trigger_slot_index: int | None = None


@dataclass(slots=True)
class SelfReactionRule:
    material: str
    trigger_slot_index: int
    phases: tuple[Phase, ...] = ()
    min_temperature: float = float("-inf")
    max_temperature: float = float("inf")
    timer_index: int | None = None
    integrity_at_most: float | None = None
    integrity_at_least: float | None = None


@dataclass(slots=True)
class WorldCommand:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReadbackRequest:
    request_id: int | None = None
    center_x: int | None = None
    center_y: int | None = None
    width: int = 1
    height: int = 1
    channels: tuple[str, ...] = ()
    observer_id: int | None = None
    label: str | None = None
    target_query_id: str | None = None
    target_dx: int = 0
    target_dy: int = 0


@dataclass(slots=True)
class ReadbackResult:
    frame_id: int
    request: ReadbackRequest
    payload: dict[str, Any]


@dataclass(slots=True)
class ForceSource:
    x: float
    y: float
    direction: tuple[float, float]
    radius: float
    strength: float
    lifetime: float
    world_x: float | None = None
    world_y: float | None = None


@dataclass(slots=True)
class EntityPlaceholder:
    entity_id: int
    x: int
    y: int
    width: int
    height: int
    material: str = "placeholder_solid"
    world_x: int | None = None
    world_y: int | None = None


@dataclass(slots=True)
class EntityState:
    entity_id: int
    x: int
    y: int
    width: int
    height: int
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    facing_xy: tuple[float, float] | None = None
    placeholder_material: str = "placeholder_solid"
    tags: tuple[str, ...] = ()
    observe_channels: tuple[str, ...] = ()
    observe_pad_cells: int = 0
    observe_width: int | None = None
    observe_height: int | None = None
    observe_label: str | None = None
    world_x: int | None = None
    world_y: int | None = None


@dataclass(slots=True)
class EntityObservationSpec:
    entity_id: int
    observe_channels: tuple[str, ...] = ()
    observe_pad_cells: int = 0
    observe_width: int | None = None
    observe_height: int | None = None
    observe_label: str | None = None


@dataclass(slots=True)
class EntityStatePatch:
    entity_id: int
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FallingIslandRecord:
    island_id: int
    bbox: tuple[int, int, int, int]
    velocity_xy: tuple[float, float]
    subcell_offset: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class PageStripeUpdate:
    axis: str
    world_start: int
    world_end: int
    buffer_start: int
    buffer_end: int
    kind: str
    cross_world_start: int | None = None
    cross_world_end: int | None = None


@dataclass(slots=True)
class ObservationTarget:
    observer_id: int
    channels: tuple[str, ...]
    center_x: int | None = None
    center_y: int | None = None
    width: int | None = None
    height: int | None = None
    entity_id: int | None = None
    pad_cells: int = 0
    label: str | None = None
    target_query_id: str | None = None
    target_dx: int = 0
    target_dy: int = 0


@dataclass(slots=True)
class TargetQuery:
    query_id: str
    anchor_filters: tuple[str, ...] = ()
    source_entity_id: int | None = None
    source_x: int | None = None
    source_y: int | None = None
    anchor_entity_id: int | None = None
    direction: str | None = None
    distance_cells: int = 0
    distance_meters: float | None = None
    distance_hint: str | None = None
    require_empty: bool = False
    search_radius: int = 0
    label: str | None = None


@dataclass(slots=True)
class ChangeIntent:
    intent_id: str
    target_query_id: str | None = None
    center_x: int | None = None
    center_y: int | None = None
    target_dx: int = 0
    target_dy: int = 0
    radius: int = 0
    material: str | None = None
    temperature_delta: float = 0.0
    velocity: tuple[float, float] | None = None
    velocity_carrier: str = "cell"
    velocity_mode: str = "add"
    require_empty: bool = False
    fallback_mode: str = "nearest_empty"
    fallback_radius: int = 0
    potency: float = 1.0
    stability: float = 1.0
    label: str | None = None


@dataclass(slots=True)
class CarrierIntent:
    intent_id: str
    kind: str
    target_query_id: str | None = None
    center_x: int | None = None
    center_y: int | None = None
    source_entity_id: int | None = None
    source_x: int | None = None
    source_y: int | None = None
    target_dx: int = 0
    target_dy: int = 0
    radius: int = 0
    material: str | None = None
    gas_species: str | None = None
    gas_amount: float = 0.0
    light_type: str | None = None
    light_strength: float = 1.0
    light_spread: float = 0.25
    force_radius: float = 0.0
    force_strength: float = 0.0
    force_lifetime: float = 0.5
    release_mode: str = "impact"
    require_empty: bool = False
    fallback_mode: str = "nearest_empty"
    fallback_radius: int = 0
    potency: float = 1.0
    stability: float = 1.0
    label: str | None = None


@dataclass(slots=True)
class ResolvedTarget:
    query_id: str
    status: str
    anchor_filters: tuple[str, ...] = ()
    direction: str | None = None
    distance_cells: int = 0
    distance_meters: float | None = None
    distance_hint: str | None = None
    label: str | None = None
    source_position: tuple[int, int] | None = None
    source_world_position: tuple[int, int] | None = None
    anchor_kind: str | None = None
    anchor_entity_id: int | None = None
    anchor_position: tuple[int, int] | None = None
    anchor_world_position: tuple[int, int] | None = None
    resolved_position: tuple[int, int] | None = None
    resolved_world_position: tuple[int, int] | None = None
    note: str | None = None


@dataclass(slots=True)
class ResolvedChangeIntent:
    intent_id: str
    status: str
    target_query_id: str | None = None
    label: str | None = None
    potency: float = 1.0
    stability: float = 1.0
    center_position: tuple[int, int] | None = None
    center_world_position: tuple[int, int] | None = None
    effective_radius: int = 0
    material: str | None = None
    temperature_delta: float = 0.0
    velocity: tuple[float, float] | None = None
    velocity_carrier: str = "cell"
    velocity_mode: str = "add"
    require_empty: bool = False
    fallback_mode: str = "nearest_empty"
    fallback_applied: bool = False
    effect_shape: str = "burst"
    effect_cells: list[tuple[int, int]] = field(default_factory=list)
    effect_bounds: tuple[int, int, int, int] | None = None
    generated_commands: list[WorldCommand] = field(default_factory=list)
    note: str | None = None


@dataclass(slots=True)
class ResolvedCarrierIntent:
    intent_id: str
    status: str
    kind: str
    target_query_id: str | None = None
    label: str | None = None
    release_mode: str = "impact"
    potency: float = 1.0
    stability: float = 1.0
    source_position: tuple[int, int] | None = None
    source_world_position: tuple[int, int] | None = None
    impact_position: tuple[int, int] | None = None
    impact_world_position: tuple[int, int] | None = None
    effective_radius: int = 0
    material: str | None = None
    gas_species: str | None = None
    gas_amount: float = 0.0
    light_type: str | None = None
    light_strength: float = 0.0
    light_spread: float = 0.25
    force_radius: float = 0.0
    force_strength: float = 0.0
    force_lifetime: float = 0.5
    direction: tuple[float, float] | None = None
    require_empty: bool = False
    fallback_mode: str = "nearest_empty"
    fallback_applied: bool = False
    effect_shape: str = "impact"
    effect_cells: list[tuple[int, int]] = field(default_factory=list)
    effect_bounds: tuple[int, int, int, int] | None = None
    generated_commands: list[WorldCommand] = field(default_factory=list)
    note: str | None = None


@dataclass(slots=True)
class ObservationResult:
    observer_id: int
    frame_id: int
    request: ReadbackRequest
    payload: dict[str, Any]


@dataclass(slots=True)
class EntityCellFeedback:
    x: int
    y: int
    present: bool
    material_id: int
    phase: int
    integrity: float
    entity_id: int


@dataclass(slots=True)
class EntityFeedback:
    entity_id: int
    bbox: tuple[int, int, int, int]
    cells: list[EntityCellFeedback] = field(default_factory=list)


@dataclass(slots=True)
class WorldFrameInput:
    submission_id: int | None = None
    focus_center: tuple[int, int] | None = None
    controller_state: Any = None
    controller_state_provided: bool = False
    entities: list[EntityState] | None = None
    entity_placeholders: list[EntityPlaceholder] | None = None
    force_sources: list[ForceSource] | None = None
    emitters: list[dict[str, Any]] | None = None
    target_queries: list[TargetQuery] = field(default_factory=list)
    change_intents: list[ChangeIntent] = field(default_factory=list)
    carrier_intents: list[CarrierIntent] = field(default_factory=list)
    observation_targets: list[ObservationTarget] = field(default_factory=list)
    readback_requests: list[ReadbackRequest] = field(default_factory=list)
    commands: list[WorldCommand] = field(default_factory=list)


@dataclass(slots=True)
class WorldFrameOutput:
    frame_id: int
    submission_id: int | None = None
    controller_state: Any = None
    consumed_readbacks: list[ReadbackResult] = field(default_factory=list)
    resolved_targets: dict[str, ResolvedTarget] = field(default_factory=dict)
    resolved_change_intents: dict[str, ResolvedChangeIntent] = field(default_factory=dict)
    resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = field(default_factory=dict)
    observations: dict[int, ObservationResult] = field(default_factory=dict)
    entity_feedback: dict[int, EntityFeedback] = field(default_factory=dict)
    paging_updates: list[PageStripeUpdate] = field(default_factory=list)
    observation_plans: list[dict[str, Any]] = field(default_factory=list)
    readback_plans: list[dict[str, Any]] = field(default_factory=list)
    bridge_upload_snapshot: dict[str, Any] = field(default_factory=dict)
    bridge_frame_snapshot: dict[str, Any] = field(default_factory=dict)
    queued_observations: int = 0
    queued_readbacks: int = 0
    queued_commands: int = 0
    placeholder_count: int = 0


@dataclass(slots=True)
class WorldFramePreview:
    controller_state: Any = None
    resolved_targets: dict[str, ResolvedTarget] = field(default_factory=dict)
    resolved_change_intents: dict[str, ResolvedChangeIntent] = field(default_factory=dict)
    resolved_carrier_intents: dict[str, ResolvedCarrierIntent] = field(default_factory=dict)
    resolved_commands: list[WorldCommand] = field(default_factory=list)
    observation_requests: list[ReadbackRequest] = field(default_factory=list)
    observation_plans: list[dict[str, Any]] = field(default_factory=list)
    readback_requests: list[ReadbackRequest] = field(default_factory=list)
    readback_plans: list[dict[str, Any]] = field(default_factory=list)
    bridge_frame_snapshot: dict[str, Any] = field(default_factory=dict)
    paging_updates: list[PageStripeUpdate] = field(default_factory=list)
    placeholder_count: int = 0
