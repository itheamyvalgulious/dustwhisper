"""Public entry point for engine-capabilities serialization.

``serialize_engine_capabilities`` composes the full capability/schema dict by
merging focused section builders in the original key order. The merged result is
byte-for-byte identical to the previous monolithic implementation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ._context import build_capabilities_context
from .meta import _capabilities_meta_section
from .entities import _capabilities_entities_section
from .views import _capabilities_views_section
from .schemas import _capabilities_schemas_section
from .tables import _capabilities_tables_section
from .control import _capabilities_control_section
from .frame import _capabilities_frame_section
from .paging import _capabilities_paging_section

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def serialize_engine_capabilities(engine: "WorldEngine") -> dict[str, Any]:
    """Serialize the engine's static capabilities and schema into a dict."""

    ctx = build_capabilities_context(engine)
    capabilities: dict[str, Any] = {}
    capabilities.update(_capabilities_meta_section(engine, ctx))
    capabilities.update(_capabilities_entities_section(engine, ctx))
    capabilities.update(_capabilities_views_section(engine, ctx))
    capabilities.update(_capabilities_schemas_section(engine, ctx))
    capabilities.update(_capabilities_tables_section(engine, ctx))
    capabilities.update(_capabilities_control_section(engine, ctx))
    capabilities.update(_capabilities_frame_section(engine, ctx))
    capabilities.update(_capabilities_paging_section(engine, ctx))
    return capabilities
