"""Explicit validation milestones preserve legacy serialized defaults."""

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
