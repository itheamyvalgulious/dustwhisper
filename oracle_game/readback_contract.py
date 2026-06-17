from __future__ import annotations

READBACK_ALLOWED_CHANNELS: tuple[str, ...] = (
    "cell",
    "ambient_temperature",
    "pressure",
    "velocity",
    "optics",
    "gas",
)

READBACK_CHANNEL_BITS: dict[str, int] = {
    channel: 1 << index
    for index, channel in enumerate(READBACK_ALLOWED_CHANNELS)
}
