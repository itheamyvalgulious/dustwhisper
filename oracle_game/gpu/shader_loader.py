"""Load GLSL compute shaders from ``oracle_game/shaders/`` with template substitution.

This mirrors :mod:`oracle_game.sim.shader_loader`.  It is duplicated in ``gpu/``
because the layering rule (``sim/`` depends on ``gpu/``, never the reverse)
forbids importing the sim loader from bridge modules.  The marker convention is
identical: ``{{NAME}}`` in a ``.comp`` file is replaced with ``str(subs[NAME])``
and a missing key raises ``KeyError`` at compile time.  Keep the two
implementations in sync.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any  # moderngl Context
from typing import Any as _Ctx

# ``oracle_game/shaders/`` — sibling of the ``gpu`` package.
SHADER_ROOT: Path = Path(__file__).resolve().parent.parent / "shaders"

# Matches a ``{{NAME}}`` substitution marker.  GLSL itself never contains
# ``{{``, so this is unambiguous.
_MARKER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

# Cache of raw file text keyed by resolved path; files are immutable at runtime.
_RAW_CACHE: dict[Path, str] = {}


def _read_raw(rel_path: str) -> str:
    """Return the raw text of ``shaders/<rel_path>`` (cached)."""
    path = (SHADER_ROOT / rel_path).resolve()
    if path not in _RAW_CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"shader not found: {path}")
        _RAW_CACHE[path] = path.read_text()
    return _RAW_CACHE[path]


def shader_source(
    rel_path: str,
    subs: dict[str, Any] | None = None,
) -> str:
    """Return the source for ``rel_path`` with ``{{NAME}}`` markers substituted.

    A marker with no matching key raises ``KeyError`` — every marker must be
    satisfied so a missing constant is caught at compile time rather than
    silently left in the source.
    """
    raw = _read_raw(rel_path)
    if subs is None:
        return raw

    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in subs:
            missing.append(name)
            return match.group(0)
        return str(subs[name])

    result = _MARKER_RE.sub(_replace, raw)
    if missing:
        # Deduplicate while preserving order for a readable error.
        raise KeyError(
            f"shader {rel_path} has unsubstituted markers: {list(dict.fromkeys(missing))}"
        )
    return result


def build_compute_shader(
    ctx: _Ctx,
    rel_path: str,
    subs: dict[str, Any] | None = None,
):
    """Compile a compute shader from a ``.comp`` file, substituting ``subs``."""
    return ctx.compute_shader(shader_source(rel_path, subs))
