"""Oscillation diagnostics must distinguish movement from stationary gaps."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SPEC = spec_from_file_location(
    "rule_controller_probe",
    Path(__file__).resolve().parents[2] / "scripts/validation/rule_controller_probe.py",
)
PROBE = module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.mark.parametrize(
    "tick,task,after,expected",
    [
        (11, "task", [0, 0], True),
        (12, "task", [0, 0], False),
        (11, "changed", [0, 0], False),
        (11, "task", [2, 0], False),
    ],
)
def test_only_a_same_task_next_tick_return_is_a_consecutive_reversal(
    tick, task, after, expected
):
    prior = ("task", [0, 0], [1, 0], 10)
    event = {"before": [1, 0], "after": after}
    assert PROBE.consecutive_reversal(prior, task, event, tick) is expected
