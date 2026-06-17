from __future__ import annotations

from dataclasses import dataclass

from oracle_game.types import PageStripeUpdate


@dataclass(slots=True)
class RingPagingWindow:
    width: int
    height: int
    active_width: int
    active_height: int
    tile_size: int = 32
    chunk_tiles: int = 8
    origin_x: int = 0
    origin_y: int = 0
    buffer_origin_x: int = 0
    buffer_origin_y: int = 0

    @property
    def chunk_size(self) -> int:
        return self.tile_size * self.chunk_tiles

    def world_to_buffer(self, world_x: int, world_y: int) -> tuple[int, int]:
        return (
            (world_x - self.origin_x + self.buffer_origin_x) % self.width,
            (world_y - self.origin_y + self.buffer_origin_y) % self.height,
        )

    def buffer_to_world(self, buffer_x: int, buffer_y: int) -> tuple[int, int]:
        return (
            self.origin_x + (buffer_x - self.buffer_origin_x) % self.width,
            self.origin_y + (buffer_y - self.buffer_origin_y) % self.height,
        )

    def active_bounds(self) -> tuple[int, int, int, int]:
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.active_width,
            self.origin_y + self.active_height,
        )

    def focus_on(self, center_x: int, center_y: int) -> list[PageStripeUpdate]:
        target_origin_x = self._snap_origin(center_x - self.active_width // 2)
        target_origin_y = self._snap_origin(center_y - self.active_height // 2)
        threshold_x = self.active_width // 4
        threshold_y = self.active_height // 4
        old_origin_x = self.origin_x
        old_origin_y = self.origin_y
        updates: list[PageStripeUpdate] = []
        x_updates: list[PageStripeUpdate] = []
        y_updates: list[PageStripeUpdate] = []
        if abs(target_origin_x - self.origin_x) > threshold_x:
            x_updates = self._advance_axis("x", target_origin_x - self.origin_x)
            updates.extend(x_updates)
        if abs(target_origin_y - self.origin_y) > threshold_y:
            y_updates = self._advance_axis("y", target_origin_y - self.origin_y)
            updates.extend(y_updates)
        final_origin_x = self.origin_x
        final_origin_y = self.origin_y
        for update in x_updates:
            if update.kind == "save":
                update.cross_world_start = old_origin_y
                update.cross_world_end = old_origin_y + self.height
            else:
                update.cross_world_start = final_origin_y
                update.cross_world_end = final_origin_y + self.height
        for update in y_updates:
            if update.kind == "save":
                update.cross_world_start = old_origin_x
                update.cross_world_end = old_origin_x + self.width
            else:
                update.cross_world_start = final_origin_x
                update.cross_world_end = final_origin_x + self.width
        return updates

    def _advance_axis(self, axis: str, delta: int) -> list[PageStripeUpdate]:
        if delta == 0:
            return []
        size = self.width if axis == "x" else self.height
        if size <= 0:
            return []
        cross_world_start = self.origin_y if axis == "x" else self.origin_x
        cross_world_end = cross_world_start + (self.height if axis == "x" else self.width)
        old_origin = self.origin_x if axis == "x" else self.origin_y
        old_buffer_origin = self.buffer_origin_x if axis == "x" else self.buffer_origin_y
        full_replace = abs(delta) >= size
        span = size if full_replace else abs(delta)
        updates: list[PageStripeUpdate] = []
        if axis == "x":
            self.origin_x += delta
            self.buffer_origin_x = (self.buffer_origin_x + delta) % self.width
            if full_replace:
                updates.extend(
                    [
                        PageStripeUpdate(
                            axis="x",
                            world_start=old_origin,
                            world_end=old_origin + self.width,
                            buffer_start=old_buffer_origin,
                            buffer_end=old_buffer_origin,
                            kind="save",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                        PageStripeUpdate(
                            axis="x",
                            world_start=self.origin_x,
                            world_end=self.origin_x + self.width,
                            buffer_start=old_buffer_origin,
                            buffer_end=old_buffer_origin,
                            kind="load",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                    ]
                )
            elif delta > 0:
                segment_start = old_buffer_origin
                segment_end = (old_buffer_origin + span) % self.width
                updates.extend(
                    [
                        PageStripeUpdate(
                            axis="x",
                            world_start=old_origin,
                            world_end=old_origin + span,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="save",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                        PageStripeUpdate(
                            axis="x",
                            world_start=old_origin + self.width,
                            world_end=old_origin + self.width + span,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="load",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                    ]
                )
            else:
                segment_start = self.buffer_origin_x
                segment_end = old_buffer_origin
                updates.extend(
                    [
                        PageStripeUpdate(
                            axis="x",
                            world_start=old_origin + self.width - span,
                            world_end=old_origin + self.width,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="save",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                        PageStripeUpdate(
                            axis="x",
                            world_start=old_origin - span,
                            world_end=old_origin,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="load",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                    ]
                )
        else:
            self.origin_y += delta
            self.buffer_origin_y = (self.buffer_origin_y + delta) % self.height
            if full_replace:
                updates.extend(
                    [
                        PageStripeUpdate(
                            axis="y",
                            world_start=old_origin,
                            world_end=old_origin + self.height,
                            buffer_start=old_buffer_origin,
                            buffer_end=old_buffer_origin,
                            kind="save",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                        PageStripeUpdate(
                            axis="y",
                            world_start=self.origin_y,
                            world_end=self.origin_y + self.height,
                            buffer_start=old_buffer_origin,
                            buffer_end=old_buffer_origin,
                            kind="load",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                    ]
                )
            elif delta > 0:
                segment_start = old_buffer_origin
                segment_end = (old_buffer_origin + span) % self.height
                updates.extend(
                    [
                        PageStripeUpdate(
                            axis="y",
                            world_start=old_origin,
                            world_end=old_origin + span,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="save",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                        PageStripeUpdate(
                            axis="y",
                            world_start=old_origin + self.height,
                            world_end=old_origin + self.height + span,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="load",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                    ]
                )
            else:
                segment_start = self.buffer_origin_y
                segment_end = old_buffer_origin
                updates.extend(
                    [
                        PageStripeUpdate(
                            axis="y",
                            world_start=old_origin + self.height - span,
                            world_end=old_origin + self.height,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="save",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                        PageStripeUpdate(
                            axis="y",
                            world_start=old_origin - span,
                            world_end=old_origin,
                            buffer_start=segment_start,
                            buffer_end=segment_end,
                            kind="load",
                            cross_world_start=cross_world_start,
                            cross_world_end=cross_world_end,
                        ),
                    ]
                )
        return updates

    def stripe_span(self, update: PageStripeUpdate) -> int:
        size = self.width if update.axis == "x" else self.height
        return min(size, abs(update.world_end - update.world_start))

    def stripe_buffer_ranges(self, update: PageStripeUpdate) -> list[tuple[int, int]]:
        size = self.width if update.axis == "x" else self.height
        span = self.stripe_span(update)
        if span <= 0 or size <= 0:
            return []
        start = update.buffer_start % size
        if span >= size:
            return [(0, size)]
        end = (start + span) % size
        if start < end:
            return [(start, end)]
        ranges = [(start, size)]
        if end > 0:
            ranges.append((0, end))
        return ranges

    def _snap_origin(self, origin: int) -> int:
        stride = max(1, self.tile_size)
        return (origin // stride) * stride
