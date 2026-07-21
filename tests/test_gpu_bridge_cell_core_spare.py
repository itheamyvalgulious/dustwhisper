from __future__ import annotations

from types import SimpleNamespace

from oracle_game.gpu.bridge import GPUBridge


class _FakeBuffer:
    def __init__(self, reserve: int) -> None:
        self.size = int(reserve)
        self.release_count = 0

    def release(self) -> None:
        self.release_count += 1


class _FakeContext:
    def __init__(self) -> None:
        self.buffers: list[_FakeBuffer] = []

    def buffer(self, *, reserve: int, dynamic: bool) -> _FakeBuffer:
        assert dynamic is True
        result = _FakeBuffer(reserve)
        self.buffers.append(result)
        return result


def test_cell_core_spare_is_lazy_sized_and_reused() -> None:
    context = _FakeContext()
    bridge = GPUBridge(ctx=context)
    world = SimpleNamespace(width=4, height=3)

    first = bridge.ensure_cell_core_spare(world)
    assert first.size == 4 * 3 * 5 * 4
    assert bridge.cell_core_spare is first
    assert "cell_core_spare" not in bridge.buffers
    assert "cell_core_spare" not in bridge.serialize_runtime_state()["buffers"]
    assert bridge.ensure_cell_core_spare(world) is first

    resized = SimpleNamespace(width=8, height=3)
    second = bridge.ensure_cell_core_spare(resized)
    assert second is not first
    assert first.release_count == 1
    assert second.size == 8 * 3 * 5 * 4

    bridge.release_resources()
    assert second.release_count == 1
    assert bridge.cell_core_spare is None


def test_attach_context_releases_private_cell_core_spare() -> None:
    old_context = _FakeContext()
    bridge = GPUBridge(ctx=old_context)
    spare = bridge.ensure_cell_core_spare(SimpleNamespace(width=2, height=2))

    bridge.attach_context(_FakeContext())

    assert spare.release_count == 1
    assert bridge.cell_core_spare is None
