from __future__ import annotations

import numpy as np


def edge_shift(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = field.shape[:2]
    pad_width = ((1, 1), (1, 1)) + ((0, 0),) * max(0, field.ndim - 2)
    padded = np.pad(field, pad_width, mode="edge")
    y0 = 1 + int(dy)
    x0 = 1 + int(dx)
    return padded[y0 : y0 + height, x0 : x0 + width, ...]


def laplace(field: np.ndarray) -> np.ndarray:
    return (
        edge_shift(field, -1, 0)
        + edge_shift(field, 1, 0)
        + edge_shift(field, 0, -1)
        + edge_shift(field, 0, 1)
        - 4.0 * field
    )


def cross_average(field: np.ndarray) -> np.ndarray:
    return 0.25 * (
        edge_shift(field, -1, 0)
        + edge_shift(field, 1, 0)
        + edge_shift(field, 0, -1)
        + edge_shift(field, 0, 1)
    )


def centered_gradient_x(field: np.ndarray) -> np.ndarray:
    return 0.5 * (edge_shift(field, 0, 1) - edge_shift(field, 0, -1))


def centered_gradient_y(field: np.ndarray) -> np.ndarray:
    return 0.5 * (edge_shift(field, 1, 0) - edge_shift(field, -1, 0))


def cross_neighbor_sum(field: np.ndarray) -> np.ndarray:
    return (
        edge_shift(field, -1, 0)
        + edge_shift(field, 1, 0)
        + edge_shift(field, 0, -1)
        + edge_shift(field, 0, 1)
    )


def advect_scalar(field: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    height, width = field.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    pos_x = grid_x + 0.5
    pos_y = grid_y + 0.5
    sample_x = np.clip(pos_x - velocity[..., 0] * dt, 0.5, width - 0.5001)
    sample_y = np.clip(pos_y - velocity[..., 1] * dt, 0.5, height - 0.5001)
    base_x = sample_x - 0.5
    base_y = sample_y - 0.5
    x0 = np.floor(base_x).astype(np.int32)
    y0 = np.floor(base_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    tx = base_x - x0
    ty = base_y - y0
    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    return (1 - tx) * (1 - ty) * a + tx * (1 - ty) * b + (1 - tx) * ty * c + tx * ty * d


def advect_vector(field: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    return np.stack([advect_scalar(field[..., index], velocity, dt) for index in range(field.shape[-1])], axis=-1)


def expand_bool_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    expanded = np.asarray(mask, dtype=np.bool_).copy()
    if expanded.size == 0 or radius <= 0:
        return expanded
    expanded.fill(False)
    padded = np.pad(np.asarray(mask, dtype=np.bool_), radius, mode="constant", constant_values=False)
    height, width = expanded.shape
    diameter = radius * 2 + 1
    for offset_y in range(diameter):
        for offset_x in range(diameter):
            expanded |= padded[offset_y : offset_y + height, offset_x : offset_x + width]
    return expanded


def tile_mask_to_cell_mask(tile_mask: np.ndarray, *, tile_size: int, width: int, height: int) -> np.ndarray:
    tile_mask_bool = np.asarray(tile_mask, dtype=np.bool_)
    active_count = int(np.count_nonzero(tile_mask_bool))
    if active_count == 0:
        return np.zeros((height, width), dtype=np.bool_)
    if active_count == int(tile_mask_bool.size):
        return np.ones((height, width), dtype=np.bool_)

    tile_size = max(1, int(tile_size))
    if active_count * 4 >= int(tile_mask_bool.size):
        expanded = np.repeat(np.repeat(tile_mask_bool, tile_size, axis=0), tile_size, axis=1)
        return expanded[:height, :width].copy()

    cell_mask = np.zeros((height, width), dtype=np.bool_)
    for tile_y, tile_x in np.argwhere(tile_mask_bool):
        x0 = int(tile_x) * tile_size
        y0 = int(tile_y) * tile_size
        x1 = min(width, x0 + tile_size)
        y1 = min(height, y0 + tile_size)
        cell_mask[y0:y1, x0:x1] = True
    return cell_mask


def tile_mask_to_gas_mask(
    tile_mask: np.ndarray,
    *,
    tile_size: int,
    gas_cell_size: int,
    width: int,
    height: int,
    gas_width: int,
    gas_height: int,
) -> np.ndarray:
    tile_mask_bool = np.asarray(tile_mask, dtype=np.bool_)
    active_count = int(np.count_nonzero(tile_mask_bool))
    if active_count == 0:
        return np.zeros((gas_height, gas_width), dtype=np.bool_)
    if active_count == int(tile_mask_bool.size):
        return np.ones((gas_height, gas_width), dtype=np.bool_)

    tile_size = max(1, int(tile_size))
    gas_cell_size = max(1, int(gas_cell_size))
    if tile_size % gas_cell_size == 0 and active_count * 4 >= int(tile_mask_bool.size):
        gas_cells_per_tile = tile_size // gas_cell_size
        expanded = np.repeat(np.repeat(tile_mask_bool, gas_cells_per_tile, axis=0), gas_cells_per_tile, axis=1)
        return expanded[:gas_height, :gas_width].copy()

    gas_mask = np.zeros((gas_height, gas_width), dtype=np.bool_)
    for tile_y, tile_x in np.argwhere(tile_mask_bool):
        x0 = int(tile_x) * tile_size
        y0 = int(tile_y) * tile_size
        x1 = min(width, x0 + tile_size)
        y1 = min(height, y0 + tile_size)
        gx0 = max(0, x0 // gas_cell_size)
        gy0 = max(0, y0 // gas_cell_size)
        gx1 = min(gas_width, (x1 + gas_cell_size - 1) // gas_cell_size)
        gy1 = min(gas_height, (y1 + gas_cell_size - 1) // gas_cell_size)
        gas_mask[gy0:gy1, gx0:gx1] = True
    return gas_mask


def unit_vector_from_name(name: str) -> tuple[int, int]:
    mapping = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    return mapping.get(name, (0, 0))
