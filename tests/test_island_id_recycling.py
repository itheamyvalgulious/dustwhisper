"""CPU-only tests for island-id allocation and high-water compaction.

Island ids are stored in 32-bit signed uniforms and float32 textures by the
GPU pipelines, so the high-water mark must stay bounded by the number of
*live* islands across allocate/settle cycles.  These tests exercise the
shared allocator without any GL context.
"""

from __future__ import annotations

from oracle_game.island_id_alloc import (
    compact_island_id_space,
    reserve_island_id_block,
)
from oracle_game.sim.gpu_collapse_labeling import _reserve_formal_component_island_ids
from oracle_game.world_cell_mutators import allocate_island_id as engine_allocate_island_id

FLOAT32_EXACT_INT_LIMIT = 1 << 24


class _FakeWorld:
    """Minimal stand-in carrying the two attributes the allocator touches."""

    def __init__(self) -> None:
        self.islands: dict[int, object] = {}
        self.next_island_id = 1

    def add_islands(self, *island_ids: int) -> None:
        for island_id in island_ids:
            self.islands[int(island_id)] = object()


def test_allocate_island_id_advances_from_high_water_mark() -> None:
    world = _FakeWorld()
    for expected in (1, 2, 3):
        island_id = engine_allocate_island_id(world)
        assert island_id == expected
        world.add_islands(island_id)
    assert world.next_island_id == 4


def test_allocate_island_id_skips_live_ids_above_mark() -> None:
    world = _FakeWorld()
    world.add_islands(5)
    world.next_island_id = 5
    assert engine_allocate_island_id(world) == 6
    assert world.next_island_id == 7


def test_reserve_block_starts_above_mark_and_live_ids() -> None:
    world = _FakeWorld()
    world.add_islands(9)
    world.next_island_id = 4
    base = reserve_island_id_block(world, 3)
    assert base == 10
    assert world.next_island_id == 13
    assert not {base, base + 1, base + 2} & set(world.islands)


def test_reserve_block_successive_reservations_are_disjoint() -> None:
    world = _FakeWorld()
    base_a = reserve_island_id_block(world, 3)
    assert base_a == 1
    # Not yet published: the mark alone keeps the next block clear.
    base_b = reserve_island_id_block(world, 3)
    assert base_b == 4
    world.add_islands(1, 2, 3)
    base_c = reserve_island_id_block(world, 3)
    assert base_c == 7


def test_allocate_island_id_respects_pending_block_reservation() -> None:
    # A block reservation bumps the mark, but its ids only appear in
    # ``world.islands`` once the runtime admission publishes them (a few
    # frames later).  Single-id allocation in between must stay above it.
    world = _FakeWorld()
    base = reserve_island_id_block(world, 3)
    assert base == 1
    island_id = engine_allocate_island_id(world)
    assert island_id == 4
    world.add_islands(*range(base, base + 3))
    world.add_islands(island_id)
    assert engine_allocate_island_id(world) == 5


def test_reserve_block_zero_capacity_reserves_nothing() -> None:
    world = _FakeWorld()
    world.next_island_id = 9
    assert reserve_island_id_block(world, 0) == 0
    assert reserve_island_id_block(world, -5) == 0
    assert world.next_island_id == 9


def test_compact_resets_mark_past_live_ids() -> None:
    world = _FakeWorld()
    world.add_islands(3, 7)
    world.next_island_id = 2**31 + 123
    compact_island_id_space(world, reservation_pending=False)
    assert world.next_island_id == 8
    assert reserve_island_id_block(world, 2) == 8


def test_compact_with_no_live_islands_resets_to_one() -> None:
    world = _FakeWorld()
    world.next_island_id = 2**40
    compact_island_id_space(world, reservation_pending=False)
    assert world.next_island_id == 1
    assert engine_allocate_island_id(world) == 1


def test_compact_deferred_while_reservation_pending() -> None:
    world = _FakeWorld()
    base = reserve_island_id_block(world, 5)
    assert base == 1
    compact_island_id_space(world, reservation_pending=True)
    assert world.next_island_id == 6
    compact_island_id_space(world, reservation_pending=False)
    assert world.next_island_id == 1


def test_pipeline_wrapper_delegates_to_shared_allocator() -> None:
    world = _FakeWorld()
    world.add_islands(1)
    base = _reserve_formal_component_island_ids(None, world, 2)
    assert base == 2
    assert world.next_island_id == 4


def test_island_ids_stay_bounded_over_alloc_settle_cycles() -> None:
    world = _FakeWorld()
    max_seen = 0
    for cycle in range(5000):
        base = reserve_island_id_block(world, 4)
        world.add_islands(*range(base, base + 4))
        extra = engine_allocate_island_id(world)
        world.add_islands(extra)
        max_seen = max(max_seen, extra)
        # Settled ids fall out of ``world.islands``; once nothing is pending
        # the mark compacts back to just past the live ids.
        for island_id in list(world.islands)[2:]:
            world.islands.pop(island_id)
        compact_island_id_space(world, reservation_pending=False)
        assert world.next_island_id <= 4
    assert max_seen < FLOAT32_EXACT_INT_LIMIT
    assert world.next_island_id < FLOAT32_EXACT_INT_LIMIT


def test_reserve_block_recovers_from_int32_overflow_high_water() -> None:
    # Regression test for the reported crash: a long session pushed
    # next_island_id past int32; compaction must bring ids back down instead
    # of keeping the stale high-water mark.
    world = _FakeWorld()
    world.next_island_id = 2**40
    compact_island_id_space(world, reservation_pending=False)
    base = reserve_island_id_block(world, 8)
    assert base == 1
    assert world.next_island_id == 9
    world.add_islands(*range(1, 9))
    compact_island_id_space(world, reservation_pending=False)
    assert reserve_island_id_block(world, 8) == 9
    assert world.next_island_id == 17
