"""Explicit validation milestones preserve legacy serialized defaults."""

from pathlib import Path

import pytest

from smartsom.config.codec import primitive
from smartsom.config.experiment import ValidationOptions
from smartsom.experiments.training_controls import ValidationControls


def test_default_serialization_and_periodic_cadence():
    controls = ValidationOptions()
    assert "updates" not in primitive(controls)
    assert [i for i in range(1, 10) if controls.due(i)] == [4, 8]


def test_nonuniform_cadence_and_selection_controls():
    controls = ValidationOptions(every_updates=None, updates=(8, 32, 64))
    assert [i for i in range(1, 65) if controls.due(i)] == [8, 32, 64]
    assert [i * 256 for i in controls.updates] == [2048, 8192, 16384]
    ValidationControls(every_updates=None, updates=controls.updates)


@pytest.mark.parametrize("updates", [(), (2, 2), (3, 2), (0, 2)])
def test_invalid_milestones(updates):
    with pytest.raises(ValueError):
        ValidationOptions(every_updates=None, updates=updates)


def test_ambiguous_or_disabled_schedule():
    with pytest.raises(ValueError):
        ValidationOptions(updates=(2, 4))
    with pytest.raises(ValueError):
        ValidationOptions(enabled=False, every_updates=None, updates=(2, 4))


@pytest.mark.parametrize("total_ticks,last_update", [(8, 2), (9, 3), (1, 1)])
def test_direct_training_preparation_bounds_milestones(total_ticks, last_update):
    from smartsom import api

    root = Path(__file__).resolve().parents[2]
    config = api.load_config(root / "configs/test/runs/train_all_ppo.yaml")
    config.training.total_ticks = total_ticks
    config.training.ticks_per_update = 4
    config.validation = ValidationOptions(every_updates=None, updates=(last_update,))
    prepared = api.prepare(config, training=True)
    assert prepared.config.training.total_ticks == total_ticks
    config.validation = ValidationOptions(
        every_updates=None, updates=(last_update + 1,)
    )
    with pytest.raises(
        ValueError, match="validation update exceeds the training budget"
    ):
        api.prepare(config, training=True)
