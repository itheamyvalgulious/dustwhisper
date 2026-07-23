"""Shared pytest infrastructure.

GPU tests need a ModernGL standalone context. On machines without a usable
GPU/driver, tests that request the ``gl_context`` / ``require_gpu`` fixtures
skip instead of failing with a raw RuntimeError.
"""

from __future__ import annotations

import pytest


def _try_create_standalone_context():
    try:
        import moderngl
    except ImportError:
        return None
    for kwargs in ({"require": 430, "backend": "egl"}, {"require": 430}):
        try:
            return moderngl.create_standalone_context(**kwargs)
        except Exception:
            continue
    return None


@pytest.fixture(scope="session")
def gl_context():
    """Session-shared ModernGL standalone context; skips when unavailable."""
    ctx = _try_create_standalone_context()
    if ctx is None:
        pytest.skip("no GPU: unable to create a ModernGL standalone context (GL >= 4.3)")
    yield ctx
    try:
        ctx.release()
    except Exception:
        pass


@pytest.fixture(scope="session")
def require_gpu(gl_context):
    """Skip the test when no GPU context can be created; returns the context."""
    return gl_context
