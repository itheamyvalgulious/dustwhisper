"""Island-id high-water allocator shared by all island-id reservation paths.

Island ids are written into 32-bit signed integer uniforms and float32
textures by the GPU pipelines, so they must stay small and bounded by the
number of live islands rather than by the number of historical allocations.

``world.next_island_id`` is a high-water mark: reservations take the lowest
block at or above the mark (never overlapping a live island id) and bump the
mark past the block, which also protects the block from single-id allocation
while its runtime records are still being published over the following frames
(see ``FormalRuntimeAdmission`` in ``gpu_collapse_incremental.py``).  All of
that is O(1) — collapse epochs routinely reserve tens of thousands of
component ids for transient fragments that never publish, so any design that
materializes those ids in a Python set costs milliseconds per frame.

Because the mark alone would grow without bound, ``compact_island_id_space``
resets it to just past the live island ids once nothing is in flight (no
pending runtime admission, no active collapse epoch), which is called once
per frame from ``CollapseSolver.advance_formal_gpu_dirty_epoch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oracle_game.world import WorldEngine


def allocate_island_id(world: "WorldEngine") -> int:
    """Allocate the lowest island id at or above the high-water mark."""
    islands = world.islands
    try:
        mark = int(world.next_island_id)
    except (TypeError, ValueError):
        mark = 1
    island_id = max(1, mark)
    while island_id in islands:
        island_id += 1
    world.next_island_id = island_id + 1
    return island_id


def reserve_island_id_block(world: "WorldEngine", count: int) -> int:
    """Reserve a contiguous block of ``count`` island ids and return its base.

    The returned block ``[base, base + count)`` never overlaps a live island
    id; bumping the high-water mark to ``base + count`` keeps later
    allocations clear of the block until its runtime records are published
    into ``world.islands``.
    """
    count = int(count)
    if count <= 0:
        return 0
    islands = world.islands
    try:
        mark = int(world.next_island_id)
    except (TypeError, ValueError):
        mark = 1
    base = max(1, mark)
    if islands:
        base = max(base, max(int(island_id) for island_id in islands) + 1)
    world.next_island_id = base + count
    return base


def compact_island_id_space(world: "WorldEngine", *, reservation_pending: bool) -> None:
    """Reset the high-water mark to just past the live island ids.

    Must only run when no reserved block is still being published
    (``reservation_pending`` covers a pending runtime admission or an active
    collapse epoch); otherwise the mark could drop below a block whose
    records have not materialized yet and later allocations would collide
    with it.
    """
    if reservation_pending:
        return
    islands = world.islands
    if islands:
        world.next_island_id = max(int(island_id) for island_id in islands) + 1
    else:
        world.next_island_id = 1
