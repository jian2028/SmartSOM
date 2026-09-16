"""Independent hand cases fixed before the buffer implementation."""

import json
from pathlib import Path

import pytest

from smartsom.domain import (
    MachineBuffers,
)

EXPECTED = json.loads(
    (
        Path(__file__).parents[2] / "data/reference/buffers/hand_expectations.json"
    ).read_text()
)


@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_strict_capacities(bad):
    with pytest.raises(ValueError):
        MachineBuffers("M1", bad, 1)
    with pytest.raises(ValueError):
        MachineBuffers("M1", 1, bad)
