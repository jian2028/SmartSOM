from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

from smartsom.domain import (
    DomainValidationError,
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
    validate_problem,
)


def operation(name="A", predecessors=(), machine="M1"):
    return Operation(name, (ProcessingMode("standard", machine, 2),), predecessors)


def workload(*jobs):
    return WorkloadInstance((Order("order", jobs),))


@pytest.mark.parametrize("duration", [0, -1, 1.5, True, False, "2", None])
def test_duration_is_a_positive_integer(duration):
    with pytest.raises(DomainValidationError, match="positive integer"):
        ProcessingMode("standard", "M1", duration)


@pytest.mark.parametrize("identifier", ["", "  ", None, 42])
@pytest.mark.parametrize(
    "make",
    [
        Machine,
        lambda value: ProcessingMode(value, "M1", 1),
        lambda value: ProcessingMode("standard", value, 1),
        lambda value: Operation(value, (ProcessingMode("standard", "M1", 1),)),
        lambda value: Job(value, (operation(),)),
        lambda value: Order(value, (Job("job", (operation(),)),)),
        lambda value: operation("A", (value,)),
    ],
)
def test_rejects_invalid_identifiers(identifier, make):
    with pytest.raises(DomainValidationError, match="non-empty string"):
        make(identifier)


@pytest.mark.parametrize(
    "make",
    [
        lambda: FactorySpec(()),
        lambda: WorkloadInstance(()),
        lambda: Order("order", ()),
        lambda: Job("job", ()),
        lambda: Operation("A", ()),
        lambda: FactorySpec(("M1",)),
        lambda: WorkloadInstance((Job("job", (operation(),)),)),
        lambda: Job("job", ("A",)),
        lambda: Operation("A", ("standard",)),
    ],
)
def test_rejects_empty_or_wrongly_typed_containers(make):
    with pytest.raises(DomainValidationError):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: FactorySpec((Machine("M1"), Machine("M1"))),
        lambda: Job("job", (operation(), operation())),
        lambda: Order(
            "order", (Job("job", (operation("A"),)), Job("job", (operation("B"),)))
        ),
        lambda: WorkloadInstance(
            (
                Order("order", (Job("a", (operation("A"),)),)),
                Order("order", (Job("b", (operation("B"),)),)),
            )
        ),
        lambda: WorkloadInstance(
            (
                Order("one", (Job("job", (operation("A"),)),)),
                Order("two", (Job("job", (operation("B"),)),)),
            )
        ),
        lambda: workload(Job("a", (operation("A"),)), Job("b", (operation("A"),))),
        lambda: Operation(
            "A",
            (ProcessingMode("standard", "M1", 1), ProcessingMode("standard", "M2", 2)),
        ),
        lambda: operation("B", ("A", "A")),
    ],
)
def test_rejects_duplicate_semantic_ids(make):
    with pytest.raises(DomainValidationError, match="duplicate"):
        make()


def test_mode_ids_are_local_and_factory_is_separate():
    data = workload(Job("a", (operation("A"), operation("B", ("A",)))))
    factory = FactorySpec((Machine("M1"),))
    validate_problem(factory, data)
    assert not hasattr(data, "machines")
    assert (
        data.operations[0].modes[0].processing_mode_id
        == data.operations[1].modes[0].processing_mode_id
    )


def test_rejects_multiple_modes_even_on_the_same_machine():
    with pytest.raises(DomainValidationError, match="exactly one"):
        Operation(
            "A", (ProcessingMode("slow", "M1", 2), ProcessingMode("fast", "M1", 1))
        )


def test_rejects_unknown_machine_at_problem_boundary():
    with pytest.raises(DomainValidationError, match="unknown machine 'M2'"):
        validate_problem(
            FactorySpec((Machine("M1"),)),
            workload(Job("job", (operation(machine="M2"),))),
        )


@pytest.mark.parametrize(
    "operations",
    [
        lambda: (operation("A"), operation("B")),
        lambda: (operation("A", ("B",)), operation("B", ("A",))),
        lambda: (operation("A", ("A",)),),
        lambda: (operation("A", ("outside",)),),
        lambda: (operation("A"), operation("B", ("A",)), operation("C", ("A",))),
        lambda: (operation("A"), operation("B", ("A",)), operation("C", ("A", "B"))),
        lambda: (operation("A"), operation("B", ("C",)), operation("C", ("B",))),
    ],
)
def test_requires_one_complete_serial_chain(operations):
    with pytest.raises(DomainValidationError):
        Job("job", operations())


def test_precedence_does_not_follow_container_order():
    chain = (operation("A"), operation("B", ("A",)), operation("C", ("B",)))
    for ordering in permutations(chain):
        job = Job("job", ordering)
        assert {op.operation_id: op.predecessor_ids for op in job.operations} == {
            "A": (),
            "B": ("A",),
            "C": ("B",),
        }


def test_caller_lists_cannot_mutate_domain_inputs():
    machines = [Machine("M1")]
    modes = [ProcessingMode("standard", "M1", 2)]
    predecessors = ["A"]
    operations = [Operation("A", modes), Operation("B", modes, predecessors)]
    jobs = [Job("job", operations)]
    orders = [Order("order", jobs)]
    factory = FactorySpec(machines)
    data = WorkloadInstance(orders)
    for original in (machines, modes, predecessors, operations, jobs, orders):
        original.clear()
    validate_problem(factory, data)
    assert data.operations[1].predecessor_ids == ("A",)
    assert len(data.operations) == 2
    with pytest.raises(FrozenInstanceError):
        data.operations[0].modes[0].nominal_ticks = 99
