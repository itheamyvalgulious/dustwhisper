"""Load GLSL compute shaders from ``oracle_game/shaders/`` with template substitution.

Phase 2 of the structural refactor moves the inline f-string GLSL out of the
``GPUxPipeline._ensure_programs`` methods into plain ``.glsl`` files.  This
loader replaces the f-string interpolation that used to live in Python.

Conversion convention
---------------------
An inline f-string shader used single-brace ``{NAME}`` for Python interpolation
and doubled braces ``{{`` / ``}}`` for literal GLSL braces.  The corresponding
``.glsl`` file uses:

* literal single braces ``{`` / ``}`` for GLSL blocks (no escaping), and
* ``{{NAME}}`` (double braces) for a substitution marker.

Because GLSL itself never contains ``{{``, this is unambiguous.  At load time
the loader replaces every ``{{NAME}}`` with ``str(subs[NAME])`` and leaves all
other text (including single braces) untouched.

Example
-------
``shaders/gas/advect_velocity.comp``::

    #version 430
    layout(local_size_x={{LOCAL_SIZE}}, local_size_y={{LOCAL_SIZE}}) in;
    layout(std430, binding=0) buffer CellCore { uint data[]; };
    ...

Python::

    from oracle_game.sim.shader_loader import build_compute_shader
    prog = build_compute_shader(ctx, "gas/advect_velocity.comp",
                                {"LOCAL_SIZE": LOCAL_SIZE})
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Any as _Ctx  # moderngl Context

# ``oracle_game/shaders/`` — sibling of the ``sim`` package.
SHADER_ROOT: Path = Path(__file__).resolve().parent.parent / "shaders"

# Matches a ``{{NAME}}`` substitution marker.  NAME may be any identifier
# (constants are typically UPPER_CASE, but a few shaders are parameterized by
# a runtime value such as ``scalar_type``).  GLSL itself never contains ``{{``,
# so this is unambiguous.
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


def shader_source(rel_path: str, subs: dict[str, Any] | None = None) -> str:
    """Return the source for ``rel_path`` with ``{{NAME}}`` markers substituted.

    ``subs`` maps marker names to values (ints/strs).  A marker with no matching
    key raises ``KeyError`` — every marker must be satisfied so a missing
    constant is caught at compile time rather than silently left in the source.
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


def build_compute_shader(ctx: _Ctx, rel_path: str, subs: dict[str, Any] | None = None):
    """Compile a compute shader from a ``.comp`` file, substituting ``subs``."""
    return ctx.compute_shader(shader_source(rel_path, subs))


def build_program(ctx: _Ctx, rel_path: str, subs: dict[str, Any] | None = None, **program_kwargs: Any):
    """Compile a linked program from a ``.glsl`` file (for vert/frag-style shaders)."""
    return ctx.program(shader_source(rel_path, subs), **program_kwargs)
