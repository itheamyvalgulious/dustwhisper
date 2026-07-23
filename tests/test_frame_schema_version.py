from __future__ import annotations

import pytest

from oracle_game.types import PageStripeUpdate, Phase
from oracle_game.world import WorldEngine
from oracle_game.world_paging import (
    PAGE_STORE_EXPORT_SCHEMA_VERSION,
    PAGE_STRIPE_SCHEMA_VERSION,
)


def _make_engine() -> WorldEngine:
    engine = WorldEngine(width=64, height=16, gas_cell_size=4)
    engine.clear_cell_region(0, 0, engine.width, engine.height)
    return engine


def _save_update() -> PageStripeUpdate:
    return PageStripeUpdate(
        axis="x", world_start=0, world_end=32, buffer_start=0, buffer_end=32, kind="save"
    )


def _load_update() -> PageStripeUpdate:
    return PageStripeUpdate(
        axis="x", world_start=0, world_end=32, buffer_start=32, buffer_end=64, kind="load"
    )


def _seed_stone(engine: WorldEngine) -> int:
    stone_id = engine.rulebook.material_id("raw_stone_solid")
    engine.material_id[4, 5] = stone_id
    engine.phase[4, 5] = int(Phase.STATIC_SOLID)
    return stone_id


def test_capture_page_stripe_stamps_schema_version_and_roundtrips_over_wire() -> None:
    src = _make_engine()
    dst = _make_engine()
    stone_id = _seed_stone(src)

    payload = src.capture_page_stripe(_save_update())

    assert payload["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION
    wire_payload = src.serialize_page_stripe_payload(payload)
    assert wire_payload["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION

    dst.apply_page_stripe(_load_update(), wire_payload, immediate=True)

    _, recorded_payload = dst.bridge_frame_page_stripes[-1]
    assert recorded_payload["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION
    assert int(recorded_payload["cell"]["material_id"][4, 5]) == stone_id


def test_store_and_load_page_stripe_preserves_schema_version() -> None:
    engine = _make_engine()
    _seed_stone(engine)

    stored_payload = engine.capture_page_stripe_to_store(_save_update())
    loaded_payload = engine.load_page_stripe(_load_update())

    assert stored_payload["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION
    assert loaded_payload is not None
    assert loaded_payload["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION


def test_apply_page_stripe_rejects_payload_missing_schema_version() -> None:
    engine = _make_engine()
    _seed_stone(engine)
    wire_payload = engine.serialize_page_stripe_payload(engine.capture_page_stripe(_save_update()))
    del wire_payload["schema_version"]

    with pytest.raises(ValueError, match=r"missing schema_version.*expected schema_version 1"):
        engine.apply_page_stripe(_load_update(), wire_payload, immediate=True)


def test_apply_page_stripe_rejects_payload_with_unsupported_schema_version() -> None:
    engine = _make_engine()
    _seed_stone(engine)
    wire_payload = engine.serialize_page_stripe_payload(engine.capture_page_stripe(_save_update()))
    wire_payload["schema_version"] = 999

    with pytest.raises(
        ValueError, match=r"unsupported page stripe schema_version 999.*expected schema_version 1"
    ):
        engine.apply_page_stripe(_load_update(), wire_payload, immediate=True)


def test_page_store_export_document_is_versioned_and_imports_as_document() -> None:
    src = _make_engine()
    dst = _make_engine()
    stone_id = _seed_stone(src)
    src.capture_page_stripe_to_store(_save_update())

    exported = src.export_page_store_entries()

    assert exported["schema_version"] == PAGE_STORE_EXPORT_SCHEMA_VERSION
    assert exported["entries"][0]["payload"]["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION

    assert dst.import_page_store_entries(exported, clear=True) == {
        "cleared": 0,
        "imported": 1,
        "stored_stripes": 1,
    }
    loaded_payload = dst.load_page_stripe(_load_update())

    assert loaded_payload is not None
    assert loaded_payload["schema_version"] == PAGE_STRIPE_SCHEMA_VERSION
    assert int(loaded_payload["cell"]["material_id"][4, 5]) == stone_id


def test_import_page_store_entries_rejects_document_missing_schema_version() -> None:
    src = _make_engine()
    dst = _make_engine()
    _seed_stone(src)
    src.capture_page_stripe_to_store(_save_update())
    exported = src.export_page_store_entries()
    del exported["schema_version"]

    with pytest.raises(ValueError, match=r"missing schema_version.*expected schema_version 1"):
        dst.import_page_store_entries(exported, clear=True)


def test_import_page_store_entries_rejects_document_with_unsupported_schema_version() -> None:
    src = _make_engine()
    dst = _make_engine()
    _seed_stone(src)
    src.capture_page_stripe_to_store(_save_update())
    exported = src.export_page_store_entries()
    exported["schema_version"] = 999

    with pytest.raises(
        ValueError,
        match=r"unsupported page store export schema_version 999.*expected schema_version 1",
    ):
        dst.import_page_store_entries(exported, clear=True)


def test_import_page_store_entries_rejects_entry_payload_missing_schema_version() -> None:
    src = _make_engine()
    dst = _make_engine()
    _seed_stone(src)
    src.capture_page_stripe_to_store(_save_update())
    entries = src.export_page_store_entries()["entries"]
    del entries[0]["payload"]["schema_version"]

    with pytest.raises(ValueError, match=r"missing schema_version.*expected schema_version 1"):
        dst.import_page_store_entries(entries, clear=True)
