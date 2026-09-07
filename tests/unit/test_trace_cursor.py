from dataclasses import FrozenInstanceError

import pytest
from test_static_engine import competition_case

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import Dispatch
from smartsom.engine import InvalidActionError, Simulator, replay


def test_incremental_trace_readers_reconstruct_the_same_episode():
    factory, workload, _ = competition_case()
    simulator = Simulator(factory, workload)
    policy = SPTPolicy()
    initial = simulator.trace_since(0)
    assert isinstance(initial, tuple)
    first_reader = list(initial)
    assert simulator.trace_since(len(first_reader)) == ()
    with pytest.raises(FrozenInstanceError):
        initial[0].simulation_time = 100
    with pytest.raises(InvalidActionError):
        simulator.step(Dispatch("missing", "standard"))
    assert simulator.trace_since(len(first_reader)) == ()
    while (context := simulator.current_decision) is not None:
        result = simulator.step(policy.select_action(context))
        first_reader.extend(simulator.trace_since(len(first_reader)))
    assert tuple(first_reader) == simulator.trace_since(0) == result.trace
    assert simulator.trace_since(len(first_reader)) == ()
    assert simulator.trace_since(len(initial)) == result.trace[len(initial) :]
    assert initial == result.trace[: len(initial)]
    assert replay(factory, workload, result.actions) == result


@pytest.mark.parametrize("cursor", [-1, 2, True, False, 0.0, 1.0, "0", None])
def test_invalid_trace_cursor_does_not_change_state(cursor):
    factory, workload, _ = competition_case()
    simulator = Simulator(factory, workload)
    before = simulator.current_decision, simulator.trace
    with pytest.raises(ValueError, match="trace cursor"):
        simulator.trace_since(cursor)
    assert (simulator.current_decision, simulator.trace) == before
