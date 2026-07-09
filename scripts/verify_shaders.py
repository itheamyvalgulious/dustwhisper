"""Verify every extracted .glsl shader renders cleanly with its stage's _SHADER_SUBS.

For each migrated GPU pipeline module, load every ``shaders/<stage>/*.comp``
(including ``_``-prefixed includes) and confirm:
  * no ``{{MARKER}}`` remains unsubstituted (every marker is in _SHADER_SUBS), and
  * no Python ``{EXPR}`` remnant (single-brace expression the loader can't match).

This is an independent check that the f-string -> .glsl extraction is complete.
Run: ``.venv/bin/python scripts/verify_shaders.py``
"""
from __future__ import annotations

import importlib
import pathlib
import re

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

SHADER_ROOT = pathlib.Path(__file__).resolve().parent.parent / "oracle_game" / "shaders"

# stage -> (module, subs). subs is either an attribute name (str) on the module,
# or a literal dict for files that pass subs per-call (no module-level _SHADER_SUBS).
STAGES = {
    "merge": ("oracle_game.sim.gpu_merge", {"LOCAL_SIZE": 8}),
    "gas": ("oracle_game.sim.gpu_gas", "_SHADER_SUBS"),
    "heat": ("oracle_game.sim.gpu_heat", "_SHADER_SUBS"),
    "optics": ("oracle_game.sim.gpu_optics", "_SHADER_SUBS"),
    "liquid": ("oracle_game.sim.gpu_liquid", "_SHADER_SUBS"),
    "motion": ("oracle_game.sim.gpu_motion", "_SHADER_SUBS"),
    "collapse": ("oracle_game.sim.gpu_collapse", "_SHADER_SUBS"),
    "reactions": ("oracle_game.sim.gpu_reactions", "_SHADER_SUBS"),
    "world_commands": ("oracle_game.sim.gpu_world_commands", "_SHADER_SUBS"),
    "placeholders": ("oracle_game.sim.gpu_placeholders", {"PASS_LOCAL_SIZE": 8, "MAX_MATERIALS": 256, "MAX_MATERIALS_MINUS_1": 255}),
    "page_stripes": ("oracle_game.sim.gpu_page_stripes", {"LOCAL_SIZE": 8, "scalar_type": "float"}),
}

MARKER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
# A single-brace Python-expr remnant: { immediately followed by an identifier-ish
# expr and closing }, NOT part of a {{...}} marker and NOT a normal GLSL block
# (GLSL blocks have a space or keyword after {, not an UPPERCASE identifier).
REMANT_RE = re.compile(r"(?<!\{)\{([A-Z_][A-Za-z0-9_ .]*[*\-/.][A-Za-z0-9_ .]*)\}(?!\})")


def main() -> int:
    from oracle_game.sim.shader_loader import shader_source

    failures = 0
    checked = 0
    for stage, (modname, subs_attr) in STAGES.items():
        mod = importlib.import_module(modname)
        subs = getattr(mod, subs_attr) if isinstance(subs_attr, str) else subs_attr
        stage_dir = SHADER_ROOT / stage
        if not stage_dir.is_dir():
            continue
        for comp in sorted(stage_dir.glob("*.comp")):
            rel = f"{stage}/{comp.name}"
            raw = comp.read_text()
            # markers present in the raw file
            markers = set(MARKER_RE.findall(raw))
            missing = {m for m in markers if subs is None or m not in (subs or {})}
            # render (only if no missing markers, else shader_source raises)
            try:
                rendered = shader_source(rel, subs)
                unsubstituted = set(MARKER_RE.findall(rendered))
                remnants = REMANT_RE.findall(rendered)
            except KeyError as ex:
                rendered = None
                unsubstituted = markers
                remnants = []
            checked += 1
            problems = []
            if unsubstituted:
                problems.append(f"unsubstituted markers: {sorted(unsubstituted)}")
            if remnants:
                problems.append(f"python-expr remnants: {remnants}")
            if problems:
                failures += 1
                print(f"FAIL {rel}: {'; '.join(problems)}")
    print(f"\nchecked {checked} shaders, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
