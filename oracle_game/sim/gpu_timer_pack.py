from __future__ import annotations

import numpy as np


def pack_u8x4(values: np.ndarray) -> np.ndarray:
    channels = np.asarray(values, dtype=np.uint8)
    if channels.shape[-1] != 4:
        raise ValueError("u8x4 timer state requires exactly four channels")
    return np.ascontiguousarray(
        channels[..., 0].astype(np.uint32)
        | (channels[..., 1].astype(np.uint32) << 8)
        | (channels[..., 2].astype(np.uint32) << 16)
        | (channels[..., 3].astype(np.uint32) << 24)
    )


def unpack_u8x4(values: np.ndarray) -> np.ndarray:
    packed = np.asarray(values, dtype=np.uint32)
    return np.stack(
        (
            packed & 0xFF,
            (packed >> 8) & 0xFF,
            (packed >> 16) & 0xFF,
            (packed >> 24) & 0xFF,
        ),
        axis=-1,
    ).astype(np.uint8)


def pack_cell_state(material_id: np.ndarray, phase: np.ndarray, flags: np.ndarray) -> np.ndarray:
    material = np.clip(np.asarray(material_id), 0, 0xFFFF).astype(np.uint32)
    phase_u32 = np.clip(np.asarray(phase), 0, 0xFF).astype(np.uint32)
    flags_u32 = np.clip(np.asarray(flags), 0, 0xFF).astype(np.uint32)
    return np.ascontiguousarray(material | (phase_u32 << 16) | (flags_u32 << 24))


def unpack_cell_state(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = np.asarray(values, dtype=np.uint32)
    return (
        (packed & 0xFFFF).astype(np.int32),
        ((packed >> 16) & 0xFF).astype(np.uint8),
        ((packed >> 24) & 0xFF).astype(np.uint8),
    )
