from __future__ import annotations

import json
from enum import Enum
from typing import Any

import numpy as np

from oracle_game.readback import READBACK_CPU_LATENCY_FRAMES, READBACK_GPU_LATENCY_FRAMES



CPU_READBACK_LATENCY_FRAMES = READBACK_CPU_LATENCY_FRAMES
GPU_READBACK_LATENCY_FRAMES = READBACK_GPU_LATENCY_FRAMES


try:  # pragma: no cover
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


_SHARED_STANDALONE_CONTEXT: Any | None = None
MAX_REACTION_LIGHT_EMITTERS = 256


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{value!r} is not JSON serializable")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, default=_json_default).encode("utf-8")


def _get_shared_standalone_context(*, require: int) -> Any | None:
    global _SHARED_STANDALONE_CONTEXT
    if moderngl is None:
        return None
    if _SHARED_STANDALONE_CONTEXT is None:
        errors: list[Exception] = []
        for kwargs in ({"require": require, "backend": "egl"}, {"require": require}):
            try:
                _SHARED_STANDALONE_CONTEXT = moderngl.create_standalone_context(**kwargs)
                break
            except Exception as exc:
                errors.append(exc)
        if _SHARED_STANDALONE_CONTEXT is None and errors:
            raise errors[-1]
    return _SHARED_STANDALONE_CONTEXT


def _render_group_tile(material: "MaterialDef", tile: int) -> np.ndarray:
    base = np.asarray(material.base_color, dtype=np.float32)
    return np.broadcast_to(np.clip(base, 0.0, 1.0), (tile, tile, 3)).copy()
