from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True)
class ActiveRegionTracker:
    world_width: int
    world_height: int
    tile_size: int = 32
    chunk_tiles: int = 8
    active_ttl_reset: int = 6
    tile_width: int = 0
    tile_height: int = 0
    chunk_width: int = 0
    chunk_height: int = 0
    active_tile_ttl: list[list[int]] | None = None
    active_chunk_mask: list[list[bool]] | None = None

    def __post_init__(self) -> None:
        self.active_ttl_reset = self._coerce_active_ttl_reset(self.active_ttl_reset)
        tile_width = max(1, (self.world_width + self.tile_size - 1) // self.tile_size)
        tile_height = max(1, (self.world_height + self.tile_size - 1) // self.tile_size)
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.chunk_width = max(1, (tile_width + self.chunk_tiles - 1) // self.chunk_tiles)
        self.chunk_height = max(1, (tile_height + self.chunk_tiles - 1) // self.chunk_tiles)
        self.active_tile_ttl = [[0 for _ in range(tile_width)] for _ in range(tile_height)]
        self.active_chunk_mask = [[False for _ in range(self.chunk_width)] for _ in range(self.chunk_height)]

    def mark_rect(self, x0: int, y0: int, x1: int, y1: int, *, tile_padding: int = 0) -> None:
        tile_x0, tile_y0, tile_x1, tile_y1 = self._tile_rect_bounds(
            x0,
            y0,
            x1,
            y1,
            tile_padding=tile_padding,
        )
        for tile_y in range(tile_y0, tile_y1):
            for tile_x in range(tile_x0, tile_x1):
                self.active_tile_ttl[tile_y][tile_x] = self.active_ttl_reset
                chunk_x = min(self.chunk_width - 1, tile_x // self.chunk_tiles)
                chunk_y = min(self.chunk_height - 1, tile_y // self.chunk_tiles)
                self.active_chunk_mask[chunk_y][chunk_x] = True

    def clear_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        tile_x0, tile_y0, tile_x1, tile_y1 = self._tile_rect_bounds(x0, y0, x1, y1)
        for tile_y in range(tile_y0, tile_y1):
            for tile_x in range(tile_x0, tile_x1):
                self.active_tile_ttl[tile_y][tile_x] = 0
        self._refresh_chunk_mask()

    def decay(self) -> None:
        for tile_y in range(self.tile_height):
            for tile_x in range(self.tile_width):
                if self.active_tile_ttl[tile_y][tile_x] > 0:
                    self.active_tile_ttl[tile_y][tile_x] -= 1
        for chunk_y in range(self.chunk_height):
            for chunk_x in range(self.chunk_width):
                active = False
                for tile_y in range(chunk_y * self.chunk_tiles, min(self.tile_height, (chunk_y + 1) * self.chunk_tiles)):
                    for tile_x in range(chunk_x * self.chunk_tiles, min(self.tile_width, (chunk_x + 1) * self.chunk_tiles)):
                        if self.active_tile_ttl[tile_y][tile_x] > 0:
                            active = True
                            break
                    if active:
                        break
                self.active_chunk_mask[chunk_y][chunk_x] = active

    def iter_active_tiles(self) -> Iterator[tuple[int, int]]:
        for tile_y in range(self.tile_height):
            for tile_x in range(self.tile_width):
                if self.active_tile_ttl[tile_y][tile_x] > 0:
                    yield tile_x, tile_y

    def _tile_rect_bounds(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        tile_padding: int = 0,
    ) -> tuple[int, int, int, int]:
        if x1 <= x0 or y1 <= y0:
            return (0, 0, 0, 0)
        tile_x0 = max(0, x0 // self.tile_size)
        tile_y0 = max(0, y0 // self.tile_size)
        tile_x1 = min(self.tile_width, (x1 + self.tile_size - 1) // self.tile_size)
        tile_y1 = min(self.tile_height, (y1 + self.tile_size - 1) // self.tile_size)
        if tile_padding > 0:
            tile_x0 = max(0, tile_x0 - tile_padding)
            tile_y0 = max(0, tile_y0 - tile_padding)
            tile_x1 = min(self.tile_width, tile_x1 + tile_padding)
            tile_y1 = min(self.tile_height, tile_y1 + tile_padding)
        return (tile_x0, tile_y0, tile_x1, tile_y1)

    def _refresh_chunk_mask(self) -> None:
        for chunk_y in range(self.chunk_height):
            for chunk_x in range(self.chunk_width):
                active = False
                for tile_y in range(chunk_y * self.chunk_tiles, min(self.tile_height, (chunk_y + 1) * self.chunk_tiles)):
                    for tile_x in range(chunk_x * self.chunk_tiles, min(self.tile_width, (chunk_x + 1) * self.chunk_tiles)):
                        if self.active_tile_ttl[tile_y][tile_x] > 0:
                            active = True
                            break
                    if active:
                        break
                self.active_chunk_mask[chunk_y][chunk_x] = active

    def __setattr__(self, name: str, value: object) -> None:
        if name == "active_ttl_reset":
            value = self._coerce_active_ttl_reset(value)
        object.__setattr__(self, name, value)

    @staticmethod
    def _coerce_active_ttl_reset(value: object) -> int:
        return max(2, int(value))
