from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from oracle_game.sim.gpu_collapse import GPUCollapsePipeline
from oracle_game.sim import gpu_collapse_incremental as incremental


@contextmanager
def _profile_pass(*_args: object, **_kwargs: object):
    yield


def _epoch(*, label_union: bool = False) -> incremental.FormalDirtyCollapseEpoch:
    return incremental.FormalDirtyCollapseEpoch(
        epoch_id=1,
        phase=0,
        started_frame_id=1,
        world_signature=(1,),
        resources=SimpleNamespace(support_ping="labels", support_pong="label_scratch"),
        x0=0,
        y0=0,
        width=64,
        height=45,
        tile_mask_name="tiles",
        support_schedule=tuple(range(8)),
        support_current="support_ping",
        support_scratch="support_pong",
        outcome_texture="outcome",
        label_tile_union_enabled=label_union,
    )


class _Phase0Pipeline:
    def __init__(self, *, v3_enabled: bool) -> None:
        self._incremental_phase_peak_v3_balance_enabled = v3_enabled
        self._incremental_jfa_four_frame_balance_enabled = True
        self._persistent_dense_tile_worklist_enabled = True
        self.stops: list[tuple[int, int]] = []

    _profile_pass = staticmethod(_profile_pass)

    def _run_formal_connected_tile_support_slice(
        self,
        _world: object,
        _resources: object,
        current: object,
        scratch: object,
        _width: int,
        _height: int,
        _tile_mask_name: str,
        _schedule: tuple[int, ...],
        start: int,
        stop: int,
    ) -> tuple[object, object]:
        self.stops.append((start, stop))
        return current, scratch


@pytest.mark.parametrize(("v3_enabled", "expected_stop"), ((False, 3), (True, 2)))
def test_phase_peak_v3_moves_only_one_phase0_support_round(
    v3_enabled: bool,
    expected_stop: int,
) -> None:
    pipeline = _Phase0Pipeline(v3_enabled=v3_enabled)
    epoch = _epoch()

    incremental._advance_phase0(pipeline, object(), epoch)

    assert pipeline.stops == [(0, expected_stop)]
    assert epoch.support_round == expected_stop
    assert epoch.phase == 1


class _LabelUnionPipeline:
    def __init__(self, *, v3_enabled: bool, fusion_enabled: bool = False) -> None:
        self._incremental_phase_peak_v3_balance_enabled = v3_enabled
        self._incremental_label_union_materialize_validation_fusion_enabled = fusion_enabled
        self._incremental_jfa_four_frame_balance_enabled = True
        self._incremental_direct_delayed_publish_enabled = True
        self.events: list[str] = []
        self.validation_kwargs: dict[str, object] = {}

    _profile_pass = staticmethod(_profile_pass)

    def _begin_formal_connected_component_label_union(
        self,
        *_args: object,
        local_components_ready: bool = False,
    ) -> tuple[int, int]:
        assert local_components_ready is False
        self.events.append("union_begin")
        return 17, 4096

    def _run_formal_connected_component_label_union_slice(
        self,
        _world: object,
        _resources: object,
        _edge_capacity: int,
        start: int,
        stop: int,
    ) -> int:
        assert (start, stop) == (0, 17)
        self.events.append("union_slice")
        return stop

    def _materialize_formal_connected_component_label_union(
        self,
        *_args: object,
    ) -> tuple[str, str]:
        self.events.append("label_materialize")
        return "labels", "label_scratch"

    def _validate_and_collect_formal_dirty_epoch_labels(
        self,
        _world: object,
        _resources: object,
        label_texture: object,
        _scratch_texture: object,
        *_args: object,
        **kwargs: object,
    ) -> tuple[object, int]:
        assert label_texture == "labels"
        self.validation_kwargs = kwargs
        self.events.append("validate_and_collect")
        return label_texture, 0

    def _publish_formal_connected_component_labels(self, *_args: object) -> None:
        self.events.append("publish_labels")


@pytest.mark.parametrize("v3_enabled", (False, True))
def test_phase_peak_v3_materializes_once_and_keeps_validation_in_phase3(
    monkeypatch: pytest.MonkeyPatch,
    v3_enabled: bool,
) -> None:
    pipeline = _LabelUnionPipeline(v3_enabled=v3_enabled)
    epoch = _epoch(label_union=True)
    world = SimpleNamespace()

    def finish_support(*_args: object, **_kwargs: object) -> None:
        pipeline.events.append("finish_support")

    monkeypatch.setattr(incremental, "_finish_support_and_resolve_outcome", finish_support)
    incremental._advance_phase2(pipeline, world, epoch)
    pipeline.events.append("frame_boundary")
    incremental._commit_formal_dirty_epoch(pipeline, world, epoch)

    assert epoch.label_union_round == epoch.label_union_round_count == 17
    assert epoch.label_union_materialized is True
    assert pipeline.events.count("label_materialize") == 1
    assert pipeline.events.index("union_slice") < pipeline.events.index("label_materialize")
    if v3_enabled:
        assert pipeline.events.index("label_materialize") < pipeline.events.index("frame_boundary")
    else:
        assert pipeline.events.index("frame_boundary") < pipeline.events.index("label_materialize")
    assert pipeline.events.index("frame_boundary") < pipeline.events.index("validate_and_collect")
    assert pipeline.events[-2:] == ["validate_and_collect", "publish_labels"]


def test_phase_peak_v3_is_default_on() -> None:
    assert GPUCollapsePipeline()._incremental_phase_peak_v3_balance_enabled is True


def test_label_union_materialize_validation_fusion_defers_without_phase2_scratch_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _LabelUnionPipeline(v3_enabled=True, fusion_enabled=True)
    epoch = _epoch(label_union=True)

    def finish_support(*_args: object, **_kwargs: object) -> None:
        pipeline.events.append("finish_support")

    monkeypatch.setattr(incremental, "_finish_support_and_resolve_outcome", finish_support)

    incremental._advance_phase2(pipeline, SimpleNamespace(), epoch)

    assert pipeline.events == ["finish_support", "union_begin", "union_slice"]
    assert epoch.label_union_materialize_validation_deferred is True
    assert epoch.label_union_materialized is False
    assert epoch.label_texture is None
    assert epoch.label_scratch is None
    assert epoch.phase == 3

    incremental._commit_formal_dirty_epoch(pipeline, SimpleNamespace(), epoch)

    assert pipeline.events == [
        "finish_support",
        "union_begin",
        "union_slice",
        "validate_and_collect",
        "publish_labels",
    ]
    assert pipeline.validation_kwargs == {"materialize_label_union": True}
    assert epoch.label_union_materialized is True
    assert epoch.label_texture == "labels"
    assert epoch.label_scratch == "label_scratch"
