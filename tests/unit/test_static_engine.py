import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path

import pytest

from smartsom.dispatch import Dispatch
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    OperationStatus,
    Order,
    ProcessingMode,
    WorkloadInstance,
)
from smartsom.engine import (
    DeadlockError,
    InvalidActionError,
    InvariantViolation,
    ReplayError,
    SimulationFinishedError,
    SimulationResult,
    Simulator,
    replay,
)
from smartsom.engine.calendar import CompletionEvent, EventCalendar
from smartsom.trace import (
    CompletionRecord,
    DecisionRecord,
    DispatchRecord,
    TerminationRecord,
)


def op(name, machine, duration, predecessor=None):
    return Operation(
        name,
        (ProcessingMode("standard", machine, duration),),
        (predecessor,) if predecessor else (),
    )


def action(operation_id):
    return Dispatch(operation_id, "standard")


def problem(*jobs):
    return FactorySpec((Machine("M1"), Machine("M2"))), WorkloadInstance(
        (Order("order", jobs),)
    )


def competition_case():
    factory, workload = problem(
        Job("A", (op("A1", "M1", 3), op("A2", "M2", 2, "A1"))),
        Job("B", (op("B1", "M1", 1), op("B2", "M2", 3, "B1"))),
    )
    return factory, workload, tuple(map(action, ("B1", "A1", "B2", "A2")))


def crossing_case():
    factory, workload = problem(
        Job("C", (op("C1", "M1", 2), op("C2", "M2", 1, "C1"))),
        Job("D", (op("D1", "M2", 3), op("D2", "M1", 2, "D1"))),
    )
    return factory, workload, tuple(map(action, ("C1", "D1", "C2", "D2")))


class SequencePolicy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def select_action(self, context):
        return next(self.actions)


class DurationPolicy:
    def __init__(self, longest=False):
        self.longest = longest

    def select_action(self, context):
        return min(
            context.candidates,
            key=lambda candidate: (
                -candidate.nominal_ticks if self.longest else candidate.nominal_ticks,
                candidate.action.operation_id,
            ),
        ).action


def assert_schedule_is_legal(workload, result):
    # Independently recompute feasibility from completed intervals, not engine state.
    expected = {operation.operation_id: operation for operation in workload.operations}
    counts = Counter(entry.operation_id for entry in result.schedule)
    assert counts == Counter({key: 1 for key in expected})
    by_operation = {entry.operation_id: entry for entry in result.schedule}
    by_machine = defaultdict(list)
    for entry in result.schedule:
        operation = expected[entry.operation_id]
        mode = operation.modes[0]
        assert (entry.processing_mode_id, entry.machine_id) == (
            mode.processing_mode_id,
            mode.machine_id,
        )
        assert entry.start_time >= 0
        assert entry.completion_time - entry.start_time == mode.nominal_ticks
        for predecessor in operation.predecessor_ids:
            assert by_operation[predecessor].completion_time <= entry.start_time
        by_machine[entry.machine_id].append(entry)
    for entries in by_machine.values():
        entries.sort(key=lambda entry: entry.start_time)
        assert all(
            left.completion_time <= right.start_time
            for left, right in zip(entries, entries[1:])
        )
    assert result.makespan == max(entry.completion_time for entry in result.schedule)
    assert [record.sequence for record in result.trace] == list(
        range(len(result.trace))
    )
    starts = [
        record.action.operation_id
        for record in result.trace
        if isinstance(record, DispatchRecord)
    ]
    finishes = [
        record.action.operation_id
        for record in result.trace
        if isinstance(record, CompletionRecord)
    ]
    assert Counter(starts) == Counter(finishes) == counts
    assert sum(isinstance(record, TerminationRecord) for record in result.trace) == 1
    assert result.trace[-1].simulation_time == result.makespan


def test_competition_matches_the_entire_hand_calculated_trace():
    factory, workload, actions = competition_case()
    result = replay(factory, workload, actions)
    assert [
        (entry.operation_id, entry.machine_id, entry.start_time, entry.completion_time)
        for entry in result.schedule
    ] == [
        ("B1", "M1", 0, 1),
        ("A1", "M1", 1, 4),
        ("B2", "M2", 1, 4),
        ("A2", "M2", 4, 6),
    ]
    assert result.makespan == 6
    assert result.trace == (
        DecisionRecord(0, 0, (action("A1"), action("B1"))),
        DispatchRecord(1, 0, action("B1"), "M1", 1),
        CompletionRecord(2, 1, action("B1"), "M1"),
        DecisionRecord(3, 1, (action("A1"), action("B2"))),
        DispatchRecord(4, 1, action("A1"), "M1", 3),
        DecisionRecord(5, 1, (action("B2"),)),
        DispatchRecord(6, 1, action("B2"), "M2", 3),
        CompletionRecord(7, 4, action("A1"), "M1"),
        CompletionRecord(8, 4, action("B2"), "M2"),
        DecisionRecord(9, 4, (action("A2"),)),
        DispatchRecord(10, 4, action("A2"), "M2", 2),
        CompletionRecord(11, 6, action("A2"), "M2"),
        TerminationRecord(12, 6),
    )
    assert_schedule_is_legal(workload, result)


def test_crossing_routes_skip_a_tick_without_a_decision():
    factory, workload, actions = crossing_case()
    simulator = Simulator(factory, workload)
    context = simulator.step(actions[0])
    assert context.simulation_time == 0
    context = simulator.step(actions[1])
    assert context.simulation_time == 3
    assert context.feasible_actions == (action("C2"), action("D2"))
    result = simulator.run(SequencePolicy(actions[2:]))
    assert [
        (entry.operation_id, entry.start_time, entry.completion_time)
        for entry in result.schedule
    ] == [("C1", 0, 2), ("D1", 0, 3), ("C2", 3, 4), ("D2", 3, 5)]
    assert result.makespan == 5
    assert not any(
        isinstance(record, DecisionRecord) and record.simulation_time == 2
        for record in result.trace
    )
    assert_schedule_is_legal(workload, result)


@pytest.mark.parametrize("case", [competition_case, crossing_case])
def test_step_run_repeat_and_replay_match(case):
    factory, workload, actions = case()
    simulator = Simulator(factory, workload)
    for selected in actions:
        stepped = simulator.step(selected)
    assert isinstance(stepped, SimulationResult)
    assert stepped == Simulator(factory, workload).run(SequencePolicy(actions))
    assert stepped == Simulator(factory, workload).run(SequencePolicy(actions))
    assert stepped == replay(factory, workload, stepped.actions)
    assert simulator.current_decision is None


def test_wrappers_call_the_public_step(monkeypatch):
    factory, workload, actions = competition_case()
    actual_step = Simulator.step
    calls = []

    def counted_step(simulator, selected):
        calls.append(selected)
        return actual_step(simulator, selected)

    monkeypatch.setattr(Simulator, "step", counted_step)
    first = Simulator(factory, workload).run(SequencePolicy(actions))
    assert calls == list(actions)
    calls.clear()
    assert replay(factory, workload, actions) == first
    assert calls == list(actions)


@pytest.mark.parametrize("case", [competition_case, crossing_case])
def test_all_tiny_legal_action_sequences_terminate(case):
    factory, workload, _ = case()
    prefixes = [()]
    completed = 0
    while prefixes:
        prefix = prefixes.pop()
        simulator = Simulator(factory, workload)
        outcome = simulator.current_decision
        for selected in prefix:
            outcome = simulator.step(selected)
        if isinstance(outcome, SimulationResult):
            completed += 1
            assert_schedule_is_legal(workload, outcome)
            assert len(outcome.actions) == len(workload.operations)
        else:
            assert len(prefix) < len(workload.operations)
            prefixes.extend(
                (*prefix, selected) for selected in outcome.feasible_actions
            )
    assert completed > 1


def test_independent_policies_use_duration_information():
    factory, workload, _ = competition_case()
    shortest = Simulator(factory, workload).run(DurationPolicy())
    longest = Simulator(factory, workload).run(DurationPolicy(longest=True))
    assert shortest.makespan == 6
    assert longest.makespan == 8
    assert shortest.actions != longest.actions
    for result in (shortest, longest):
        assert_schedule_is_legal(workload, result)


@pytest.mark.parametrize(
    ("selected", "reason"),
    [
        (action("A2"), "predecessor"),
        (action("missing"), "unknown operation"),
        (Dispatch("B1", "missing"), "unknown processing mode"),
        (None, "expected Dispatch"),
        (Dispatch([], "standard"), "unknown operation"),
    ],
)
def test_invalid_actions_are_atomic_and_recoverable(selected, reason):
    factory, workload, actions = competition_case()
    simulator = Simulator(factory, workload)
    context, trace = simulator.current_decision, simulator.trace
    with pytest.raises(InvalidActionError, match=reason) as error:
        simulator.step(selected)
    assert error.value.simulation_time == 0
    assert error.value.action == selected
    assert simulator.current_decision is context
    assert simulator.trace == trace
    assert simulator.run(SequencePolicy(actions)) == replay(factory, workload, actions)


def test_busy_machine_and_started_operation_are_rejected():
    factory, workload = problem(
        Job("A", (op("A1", "M1", 3),)),
        Job("B", (op("B1", "M1", 1),)),
        Job("C", (op("C1", "M2", 5),)),
    )
    simulator = Simulator(factory, workload)
    context = simulator.step(action("A1"))
    assert context.simulation_time == 0
    trace = simulator.trace
    for selected, reason in (
        (action("B1"), "machine is busy"),
        (action("A1"), "already started"),
    ):
        with pytest.raises(InvalidActionError, match=reason):
            simulator.step(selected)
        assert simulator.current_decision is context
        assert simulator.trace == trace
    context = simulator.step(action("C1"))
    assert context.simulation_time == 3
    with pytest.raises(InvalidActionError, match="already started"):
        simulator.step(action("A1"))
    result = simulator.step(action("B1"))
    assert_schedule_is_legal(workload, result)


def test_terminal_calls_and_replay_length_errors_are_explicit():
    factory, workload, actions = competition_case()
    simulator = Simulator(factory, workload)
    result = simulator.run(SequencePolicy(actions))
    with pytest.raises(SimulationFinishedError):
        simulator.step(actions[-1])
    with pytest.raises(SimulationFinishedError):
        simulator.run(SequencePolicy(actions))
    assert simulator.trace == result.trace
    for shortened in ((), actions[:-1]):
        with pytest.raises(ReplayError, match="before completion"):
            replay(factory, workload, shortened)
    with pytest.raises(ReplayError, match="extra action"):
        replay(factory, workload, (*actions, actions[-1]))
    with pytest.raises(InvalidActionError):
        replay(factory, workload, (action("A2"),))


def test_public_snapshots_are_immutable_and_reads_do_not_emit_trace():
    factory, workload, actions = crossing_case()
    simulator = Simulator(factory, workload)
    original = simulator.current_decision
    initial_trace = simulator.trace
    for _ in range(3):
        assert simulator.current_decision is original
        assert simulator.trace == initial_trace
    targets = [
        (original, "simulation_time", 99),
        (original.operations[0], "status", OperationStatus.COMPLETED),
        (original.machines[0], "operation_id", "bogus"),
        (original.candidates[0], "nominal_ticks", 99),
        (original.feasible_actions[0], "operation_id", "bogus"),
    ]
    for target, field, value in targets:
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, value)
    simulator.step(actions[0])
    assert all(state.status == OperationStatus.PENDING for state in original.operations)
    result = simulator.run(SequencePolicy(actions[1:]))
    for target, field, value in (
        (result, "makespan", 0),
        (result.schedule[0], "completion_time", 0),
        (result.trace[0], "simulation_time", 99),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, field, value)


def test_permuting_all_input_collections_preserves_the_trace():
    factory, workload, _ = competition_case()
    workload = replace(
        workload,
        orders=(*workload.orders, Order("extra", (Job("E", (op("E1", "M2", 1),)),))),
    )
    first = Simulator(factory, workload).run(DurationPolicy())
    reordered = WorkloadInstance(
        tuple(
            replace(
                order,
                jobs=tuple(
                    replace(job, operations=tuple(reversed(job.operations)))
                    for job in reversed(order.jobs)
                ),
            )
            for order in reversed(workload.orders)
        )
    )
    reordered_factory = FactorySpec(tuple(reversed(factory.machines)))
    assert replay(reordered_factory, reordered, first.actions) == first
    assert Simulator(reordered_factory, reordered).run(DurationPolicy()) == first


def test_completion_order_does_not_depend_on_dispatch_insertion_order():
    factory, workload = problem(
        Job("A", (op("A1", "M1", 2),)), Job("B", (op("B1", "M2", 2),))
    )
    completions = []
    for actions in permutations((action("A1"), action("B1"))):
        result = replay(factory, workload, actions)
        completions.append(
            tuple(
                record
                for record in result.trace
                if isinstance(record, CompletionRecord)
            )
        )
    assert completions[0] == completions[1]
    assert [record.action.operation_id for record in completions[0]] == ["A1", "B1"]


def test_calendar_orders_time_before_semantic_key():
    events = (
        CompletionEvent(3, "A", "standard", "M1"),
        CompletionEvent(1, "Z", "standard", "M2"),
        CompletionEvent(3, "B", "standard", "M2"),
    )
    for ordering in permutations(events):
        calendar = EventCalendar()
        for event in ordering:
            calendar.schedule(event)
        assert calendar.next_time == 1
        assert tuple(calendar.pop() for _ in events) == (
            events[1],
            events[0],
            events[2],
        )
        assert calendar.next_time is None


def test_lost_completion_is_detected_at_the_transition(monkeypatch):
    factory, workload, actions = crossing_case()
    simulator = Simulator(factory, workload)
    monkeypatch.setattr(EventCalendar, "schedule", lambda self, event: None)
    with pytest.raises(InvariantViolation, match="completion events"):
        simulator.step(actions[0])


def test_the_published_feasible_view_is_authoritative(monkeypatch):
    import smartsom.engine.simulator as engine_module

    original = engine_module.build_decision

    def only_b1(*args):
        context = original(*args)
        return replace(
            context,
            candidates=tuple(
                candidate
                for candidate in context.candidates
                if candidate.action.operation_id == "B1"
            ),
        )

    monkeypatch.setattr(engine_module, "build_decision", only_b1)
    factory, workload, _ = competition_case()
    simulator = Simulator(factory, workload)
    assert simulator.current_decision.feasible_actions == (action("B1"),)
    original_trace = simulator.trace
    with pytest.raises(InvalidActionError, match="excluded from the current"):
        simulator.step(action("A1"))
    assert simulator.trace == original_trace


def test_one_machine_can_process_successive_operations_of_the_same_job():
    factory = FactorySpec((Machine("M1"),))
    workload = WorkloadInstance(
        (Order("order", (Job("job", (op("A", "M1", 2), op("B", "M1", 3, "A"))),)),)
    )
    simulator = Simulator(factory, workload)
    context = simulator.step(action("A"))
    assert context.simulation_time == 2
    assert context.feasible_actions == (action("B"),)
    result = simulator.step(action("B"))
    assert result.makespan == 5
    assert_schedule_is_legal(workload, result)


def test_deadlock_reports_time_and_unfinished_ids(monkeypatch):
    # Inject a broken feasibility projection without corrupting domain inputs.
    import smartsom.engine.simulator as engine_module

    original = engine_module.build_decision
    monkeypatch.setattr(
        engine_module,
        "build_decision",
        lambda *args: replace(original(*args), candidates=()),
    )
    factory, workload, _ = competition_case()
    with pytest.raises(
        DeadlockError, match=r"tick 0; unfinished=\['A1', 'A2', 'B1', 'B2'\]"
    ):
        Simulator(factory, workload)


def test_hash_seed_does_not_change_canonical_results():
    test_file = str(Path(__file__).resolve())
    code = f"""
import json, runpy
from dataclasses import asdict, replace
from smartsom.engine import replay
from smartsom.domain import FactorySpec, WorkloadInstance
helpers = runpy.run_path({test_file!r})
factory, workload, actions = helpers['competition_case']()
machines = {{machine.machine_id: machine for machine in factory.machines}}
factory = FactorySpec(tuple(machines[key] for key in set(machines)))
order = workload.orders[0]
jobs = {{job.job_id: job for job in order.jobs}}
workload = WorkloadInstance((replace(order, jobs=tuple(jobs[key] for key in set(jobs))),))
print(json.dumps(asdict(replay(factory, workload, actions)), sort_keys=True))
"""
    outputs = []
    for seed in ("1", "17", "321"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(
            subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        )
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0])["makespan"] == 6


def test_base_imports_do_not_attempt_optional_framework_imports():
    code = """
import importlib.abc
import sys
class RejectOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gymnasium', 'ray', 'pettingzoo', 'torch', 'ortools', 'pyjobshop', 'wandb', 'pydantic', 'yaml'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectOptionalImports())
import smartsom
import smartsom.domain
import smartsom.dispatch
import smartsom.engine
import smartsom.trace
import smartsom.algorithms
import smartsom.algorithms.solver
import smartsom.algorithms.pyjobshop
"""
    subprocess.run([sys.executable, "-c", code], check=True)
