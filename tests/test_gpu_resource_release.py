"""Consistency tests for GPU resource release paths.

Every ``GPU*Resources`` dataclass owns GL objects (textures, buffers,
framebuffers) that must be freed exactly once when its pipeline is released.
Release paths discover the fields via ``dataclasses.fields`` (see
:func:`oracle_game.sim.gpu_base.release_resource_fields`) instead of a
hand-written field list, and these tests mirror that discovery: every field
holding a mock must see ``release()`` called exactly once.  Adding a new
resource field to one of these dataclasses therefore extends the tests
automatically — if a release path stops covering it, they go red.
"""

from __future__ import annotations

import dataclasses
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from oracle_game.sim import (
    gpu_collapse_stages,
    gpu_heat_resources,
    gpu_liquid_resources,
    gpu_motion_resources,
    gpu_reactions_transient,
)
from oracle_game.sim.gpu_base import release_resource_fields
from oracle_game.sim.gpu_collapse import GPUCollapseResources
from oracle_game.sim.gpu_gas import GPUGasPipeline, GPUGasResources
from oracle_game.sim.gpu_heat import GPUHeatResources
from oracle_game.sim.gpu_liquid import GPULiquidResources
from oracle_game.sim.gpu_motion import GPUMotionResources
from oracle_game.sim.gpu_optics import GPUOpticsPipeline, GPUOpticsResources
from oracle_game.sim.gpu_placeholders import (
    GPUPlaceholderPipeline,
    GPUPlaceholderResources,
)
from oracle_game.sim.gpu_reactions import GPUReactionResources
from oracle_game.sim.gpu_world_commands import (
    GPUWorldCommandPipeline,
    GPUWorldCommandResources,
)

ALL_RESOURCE_CLASSES = (
    GPUCollapseResources,
    GPUHeatResources,
    GPUMotionResources,
    GPULiquidResources,
    GPUGasResources,
    GPUOpticsResources,
    GPUPlaceholderResources,
    GPUReactionResources,
    GPUWorldCommandResources,
)


def _mock_resources(cls):
    """Build *cls* with a MagicMock per GL-object field.

    Fields typed ``Any`` (the project convention for GL objects) get mocks;
    plain-value fields (``signature`` tuples, counters, flags) get their
    dataclass default or an empty tuple, matching what the release filter
    must skip.
    """
    kwargs = {}
    for field in dataclasses.fields(cls):
        if "Any" in str(field.type):
            kwargs[field.name] = MagicMock(name=f"{cls.__name__}.{field.name}")
        elif field.default is not dataclasses.MISSING:
            kwargs[field.name] = field.default
        else:
            kwargs[field.name] = ()
    return cls(**kwargs)


def _assert_every_mock_field_released_once(resources) -> int:
    checked = 0
    for field in dataclasses.fields(resources):
        value = getattr(resources, field.name)
        if isinstance(value, MagicMock):
            value.release.assert_called_once_with(), field.name
            checked += 1
    assert checked > 0, f"{type(resources).__name__}: no mock fields checked"
    return checked


@pytest.mark.parametrize("cls", ALL_RESOURCE_CLASSES, ids=lambda c: c.__name__)
def test_release_resource_fields_releases_every_releasable_field(cls) -> None:
    resources = _mock_resources(cls)
    release_resource_fields(resources)
    _assert_every_mock_field_released_once(resources)


def test_release_resource_fields_skips_none_and_plain_values() -> None:
    resources = _mock_resources(GPUCollapseResources)
    resources.support_u8_ping = None
    resources.support_u8_pong = None
    release_resource_fields(resources)
    _assert_every_mock_field_released_once(resources)


def test_release_resource_fields_logs_and_continues_on_failure(caplog) -> None:
    resources = _mock_resources(GPUGasResources)
    resources.pressure_ping.release.side_effect = RuntimeError("gl gone")
    with caplog.at_level(logging.DEBUG, logger="oracle_game.sim.gpu_base"):
        release_resource_fields(resources)
    _assert_every_mock_field_released_once(resources)
    records = [
        record
        for record in caplog.records
        if record.name == "oracle_game.sim.gpu_base" and record.exc_info is not None
    ]
    assert len(records) == 1
    assert "pressure_ping" in records[0].getMessage()


def test_collapse_release_releases_every_field() -> None:
    pipeline = MagicMock(name="collapse_pipeline")
    resources = _mock_resources(GPUCollapseResources)
    pipeline.resources = resources
    gpu_collapse_stages.release(pipeline)
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_heat_release_releases_every_field() -> None:
    pipeline = MagicMock(name="heat_pipeline")
    resources = _mock_resources(GPUHeatResources)
    pipeline.resources = resources
    gpu_heat_resources.release(pipeline)
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_motion_release_releases_every_field() -> None:
    pipeline = MagicMock(name="motion_pipeline")
    resources = _mock_resources(GPUMotionResources)
    pipeline.resources = resources
    gpu_motion_resources.release(pipeline)
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_liquid_release_releases_every_field() -> None:
    pipeline = MagicMock(name="liquid_pipeline")
    resources = _mock_resources(GPULiquidResources)
    pipeline.resources = resources
    gpu_liquid_resources.release(pipeline)
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_reactions_release_releases_every_field() -> None:
    pipeline = MagicMock(name="reactions_pipeline")
    resources = _mock_resources(GPUReactionResources)
    pipeline.resources = resources
    gpu_reactions_transient.release(pipeline)
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_optics_release_releases_every_field() -> None:
    pipeline = GPUOpticsPipeline.__new__(GPUOpticsPipeline)
    resources = _mock_resources(GPUOpticsResources)
    pipeline.resources = resources
    pipeline.release()
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_gas_release_releases_every_field() -> None:
    pipeline = GPUGasPipeline.__new__(GPUGasPipeline)
    resources = _mock_resources(GPUGasResources)
    pipeline.resources = resources
    pipeline.release()
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def test_placeholders_release_releases_every_field() -> None:
    pipeline = GPUPlaceholderPipeline.__new__(GPUPlaceholderPipeline)
    resources = _mock_resources(GPUPlaceholderResources)
    pipeline.resources = resources
    pipeline.release()
    _assert_every_mock_field_released_once(resources)
    assert pipeline.resources is None


def _bare_world_command_pipeline(**overrides) -> GPUWorldCommandPipeline:
    pipeline = GPUWorldCommandPipeline.__new__(GPUWorldCommandPipeline)
    pipeline.resources = None
    pipeline.programs = {}
    pipeline._thread_contexts = {}
    pipeline._thread_resources = {}
    pipeline._thread_programs = {}
    pipeline._ephemeral_context_keys = set()
    pipeline._active_thread_id = None
    for name, value in overrides.items():
        setattr(pipeline, name, value)
    return pipeline


def test_world_commands_release_releases_every_field() -> None:
    resources = _mock_resources(GPUWorldCommandResources)
    pipeline = _bare_world_command_pipeline(_thread_resources={1: resources})
    pipeline.release()
    _assert_every_mock_field_released_once(resources)


def test_world_commands_release_context_key_releases_every_field() -> None:
    resources = _mock_resources(GPUWorldCommandResources)
    pipeline = _bare_world_command_pipeline(_thread_resources={7: resources})
    pipeline._release_context_key(7)
    _assert_every_mock_field_released_once(resources)
    assert pipeline._thread_resources == {}


def test_world_commands_ensure_resources_releases_previous_fields() -> None:
    old_resources = _mock_resources(GPUWorldCommandResources)
    pipeline = _bare_world_command_pipeline(
        resources=old_resources,
        _active_thread_id=1,
    )
    pipeline._active_context = MagicMock(return_value=MagicMock(name="ctx"))
    world = MagicMock(name="world")
    world.width = 4
    world.height = 4
    world.gas_width = 2
    world.gas_height = 2
    world.gas_concentration = np.zeros((3, 2, 2), dtype=np.float32)
    pipeline._ensure_resources(world)
    _assert_every_mock_field_released_once(old_resources)
    assert pipeline.resources is not old_resources
    assert pipeline._thread_resources[1] is pipeline.resources
