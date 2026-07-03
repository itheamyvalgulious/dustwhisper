from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oracle_game.types import ReadbackRequest, ReadbackResult


READBACK_CPU_LATENCY_FRAMES = 1
READBACK_GPU_LATENCY_FRAMES = 2


@dataclass(slots=True)
class PBOSlot:
    slot_index: int
    frame_id: int = -1
    ready_frame_id: int = -1
    min_poll_frame_id: int = -1
    latency_frames: int = READBACK_CPU_LATENCY_FRAMES
    request: ReadbackRequest | None = None
    payload: dict[str, Any] | None = None


class PBOReadbackRing:
    """Fixed-capacity delayed readback ring with PBO-style poll semantics."""

    def __init__(self, slots: int = 2, *, latency_frames: int = READBACK_CPU_LATENCY_FRAMES) -> None:
        if slots < 2:
            raise ValueError("readback ring needs at least two slots")
        if latency_frames < 1:
            raise ValueError("readback latency must be at least one frame")
        self.slots = [PBOSlot(slot_index=index) for index in range(slots)]
        self.write_index = 0
        self.read_index = 1 % slots
        self.latency_frames = int(latency_frames)

    def queue(
        self,
        frame_id: int,
        request: ReadbackRequest,
        payload: dict[str, Any],
        *,
        latency_frames: int | None = None,
    ) -> bool:
        slot: PBOSlot | None = None
        slot_count = len(self.slots)
        for offset in range(slot_count):
            candidate = self.slots[(self.write_index + offset) % slot_count]
            if candidate.frame_id < 0 and candidate.request is None and candidate.payload is None:
                slot = candidate
                break
        if slot is None:
            return False
        resolved_latency = self.latency_frames if latency_frames is None else int(latency_frames)
        if resolved_latency < 1:
            raise ValueError("readback latency must be at least one frame")
        slot.frame_id = frame_id
        slot.ready_frame_id = frame_id + READBACK_CPU_LATENCY_FRAMES
        slot.min_poll_frame_id = frame_id + resolved_latency
        slot.latency_frames = resolved_latency
        slot.request = request
        slot.payload = payload
        self.read_index = slot.slot_index
        self.write_index = (slot.slot_index + 1) % len(self.slots)
        return True

    def poll(self, current_frame_id: int) -> ReadbackResult | None:
        ready = [
            slot
            for slot in self.slots
            if slot.frame_id >= 0
            and slot.request is not None
            and slot.payload is not None
            and slot.min_poll_frame_id >= 0
            and slot.min_poll_frame_id <= current_frame_id
        ]
        if not ready:
            return None
        slot = min(ready, key=lambda item: (item.frame_id, item.slot_index))
        result = ReadbackResult(frame_id=slot.frame_id, request=slot.request, payload=slot.payload)
        slot.frame_id = -1
        slot.ready_frame_id = -1
        slot.min_poll_frame_id = -1
        slot.latency_frames = self.latency_frames
        slot.request = None
        slot.payload = None
        self.read_index = slot.slot_index
        return result
