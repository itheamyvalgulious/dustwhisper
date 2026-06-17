from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from oracle_game.types import PageStripeUpdate


def _clone_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_payload(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_payload(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_payload(child) for child in value)
    return value


@dataclass(frozen=True, slots=True)
class StoredStripeKey:
    axis: str
    world_start: int
    world_end: int
    cross_world_start: int = 0
    cross_world_end: int = 0


@runtime_checkable
class PageStore(Protocol):
    def save(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None: ...

    def load(self, update: PageStripeUpdate) -> dict[str, Any] | None: ...

    def has(self, update: PageStripeUpdate) -> bool: ...

    def keys(self) -> list[StoredStripeKey]: ...

    def stored_count(self) -> int: ...

    def clear(self) -> None: ...


class InMemoryPageStore(PageStore):
    def __init__(self) -> None:
        self._payloads: dict[StoredStripeKey, dict[str, Any]] = {}

    def save(self, update: PageStripeUpdate, payload: dict[str, Any]) -> None:
        self._payloads[self._key(update)] = _clone_payload(payload)

    def load(self, update: PageStripeUpdate) -> dict[str, Any] | None:
        payload = self._payloads.get(self._key(update))
        if payload is None:
            payload = self._payloads.get(self._legacy_key(update))
        if payload is None:
            return None
        return _clone_payload(payload)

    def has(self, update: PageStripeUpdate) -> bool:
        return self._key(update) in self._payloads or self._legacy_key(update) in self._payloads

    def keys(self) -> list[StoredStripeKey]:
        return sorted(
            self._payloads,
            key=lambda key: (
                str(key.axis),
                int(key.world_start),
                int(key.world_end),
                int(key.cross_world_start),
                int(key.cross_world_end),
            ),
        )

    def stored_count(self) -> int:
        return len(self._payloads)

    def clear(self) -> None:
        self._payloads.clear()

    def _key(self, update: PageStripeUpdate) -> StoredStripeKey:
        return StoredStripeKey(
            axis=update.axis,
            world_start=update.world_start,
            world_end=update.world_end,
            cross_world_start=0 if update.cross_world_start is None else int(update.cross_world_start),
            cross_world_end=0 if update.cross_world_end is None else int(update.cross_world_end),
        )

    def _legacy_key(self, update: PageStripeUpdate) -> StoredStripeKey:
        return StoredStripeKey(
            axis=update.axis,
            world_start=update.world_start,
            world_end=update.world_end,
            cross_world_start=0,
            cross_world_end=0,
        )
