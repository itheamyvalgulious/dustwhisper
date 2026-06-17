from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oracle_game.types import ReadbackRequest, ReadbackResult


@dataclass(slots=True)
class PBOSlot:
    slot_index: int
    frame_id: int = -1
    request: ReadbackRequest | None = None
    payload: dict[str, Any] | None = None


class PBOReadbackRing:
    """CPU-side stand-in for ping/pong PBO scheduling semantics."""

    def __init__(self, slots: int = 2) -> None:
        if slots < 2:
            raise ValueError("readback ring needs at least two slots")
        self.slots = [PBOSlot(slot_index=index) for index in range(slots)]
        self.write_index = 0
        self.read_index = 1 % slots

    def queue(self, frame_id: int, request: ReadbackRequest, payload: dict[str, Any]) -> bool:
        slot: PBOSlot | None = None
        slot_count = len(self.slots)
        for offset in range(slot_count):
            candidate = self.slots[(self.write_index + offset) % slot_count]
            if candidate.frame_id < 0 and candidate.request is None and candidate.payload is None:
                slot = candidate
                break
        if slot is None:
            return False
        slot.frame_id = frame_id
        slot.request = request
        slot.payload = payload
        self.read_index = slot.slot_index
        self.write_index = (slot.slot_index + 1) % len(self.slots)
        return True

    def poll(self, current_frame_id: int) -> ReadbackResult | None:
        ready = [
            slot
            for slot in self.slots
            if slot.frame_id >= 0 and slot.request is not None and slot.payload is not None and slot.frame_id < current_frame_id
        ]
        if not ready:
            return None
        slot = min(ready, key=lambda item: (item.frame_id, item.slot_index))
        result = ReadbackResult(frame_id=slot.frame_id, request=slot.request, payload=slot.payload)
        slot.frame_id = -1
        slot.request = None
        slot.payload = None
        self.read_index = slot.slot_index
        return result
