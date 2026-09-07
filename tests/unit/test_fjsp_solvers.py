import ast
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_experiments import ROOT, json_file, run_path
from test_experiments import bundle as bundle
from test_fjsp_engine import FAST, flexible_case
from test_static_engine import assert_schedule_is_legal

from smartsom.algorithms import SPTPolicy
from smartsom.algorithms.pyjobshop import PyJobShopAdapter
from smartsom.algorithms.solver import ScheduleSolution, SolveRequest, SolverStatus
from smartsom.config import resolve_run
from smartsom.config.codec import digest
from smartsom.domain import ProcessingMode, ScheduledOperation
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.experiments import RunFailedError, run_one
from smartsom.workloads import import_fjs


@pytest.mark.parametrize(
    "name,optimum,operations,modes",
    [("pyjobshop_fjsp", 6, 9, 27), ("mk01", 40, 55, 115)],
)
def test_fixed_external_data_and_reference_intervals(name, optimum, operations, modes):
    directory = ROOT / "data/reference" / name
    source = json_file(directory, "sources.json")
    for filename, sha in source["snapshots"].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == sha
    resolved = resolve_run(ROOT / f"configs/runs/{name}_spt.yaml")
    assert len(resolved.workload.operations) == operations
    assert sum(len(op.modes) for op in resolved.workload.operations) == modes
    assert source["workload_sha256"] == resolved.workload_sha256
    if name == "mk01":
        imported = import_fjs(directory / "Mk01.fjs", instance_id="mk01")
        assert imported.workload == resolved.workload
        assert imported.provenance.source_sha256 == source["source_sha256"]
        bounds = json_file(directory, "scheduleopt_bounds.json")
        assert bounds["lower_bound"] == bounds["upper_bound"] == optimum
        # Independent token decoder for the valid frozen fixture, not the ingress parser.
        routes = []
        for line in (directory / "Mk01.fjs").read_text().splitlines()[1:]:
            if not line.strip():
                continue
            tokens = iter(map(int, line.split()))
            routes.append(
                [
                    [(next(tokens), next(tokens)) for _ in range(next(tokens))]
                    for _ in range(next(tokens))
                ]
            )
            assert list(tokens) == []
    else:
        cell = ast.parse((directory / "data_cell.txt").read_text())
        data = next(
            ast.literal_eval(node.value)
            for node in cell.body
            if isinstance(node, ast.Assign) and node.targets[0].id == "data"
        )
        assert json.loads(json.dumps(data)) == json_file(directory, "data.json")
        routes = [
            [[(m + 1, d) for d, m in alternatives] for alternatives in job]
            for job in data
        ]
    operations_by_id = {op.operation_id: op for op in resolved.workload.operations}
    for j, job in enumerate(routes, 1):
        for i, pairs in enumerate(job, 1):
            op = operations_by_id[f"{name}/job_{j}/op_{i}"]
            assert sorted(
                (int(m.machine_id[1:]), m.nominal_ticks) for m in op.modes
            ) == sorted(pairs)
    reference = json_file(directory, "schedule.json")
    schedule = tuple(ScheduledOperation(**entry) for entry in reference["schedule"])
    assert digest(schedule) == source["schedule_sha256"]
    direct = json_file(directory, "direct_solver_result.json")
    for row in direct["tasks"]:
        entry = next(
            e
            for e in schedule
            if e.operation_id
            == f"{name}/job_{row['job'] + 1}/op_{row['operation'] + 1}"
        )
        assert (entry.machine_id, entry.start_time, entry.completion_time) == (
            f"M{row['resources'][0] + 1}",
            row["start"],
            row["end"],
        )
        assert entry.completion_time - entry.start_time == row["duration"]
    result = replay_schedule(resolved.factory, resolved.workload, reversed(schedule))
    assert result.schedule == schedule
    assert result.makespan == direct["objective"] == direct["bound"] == optimum
    assert replay(resolved.factory, resolved.workload, result.actions) == result
    assert_schedule_is_legal(resolved.workload, result)
    spt = Simulator(resolved.factory, resolved.workload).run(SPTPolicy())
    assert_schedule_is_legal(resolved.workload, spt)
    assert (
        replay_schedule(resolved.factory, resolved.workload, spt.schedule).schedule
        == spt.schedule
    )


@pytest.fixture
def fake_backend(monkeypatch):
    tasks = [
        SimpleNamespace(
            present=True,
            idle=0,
            breaks=0,
            mode=mode,
            resources=[resource],
            start=start,
            end=end,
        )
        for mode, resource, start, end in [
            (0, 1, 0, 1),
            (2, 0, 1, 3),
            (3, 1, 1, 3),
            (4, 0, 3, 4),
        ]
    ]

    class Model:
        def __init__(self):
            self.modes = []

        def add_machine(self, name):
            return name

        def add_job(self, name):
            return name

        def add_task(self, *, job, name):
            return name

        def add_mode(self, task, machine, duration):
            self.modes.append((task, machine, duration))

        def add_end_before_start(self, first, second):
            assert (first, second) in [("A1", "A2"), ("B1", "B2")]

        def solve(self, **kwargs):
            assert self.modes == [
                ("A1", "M2", 1),
                ("A1", "M1", 4),
                ("A2", "M1", 2),
                ("B1", "M2", 2),
                ("B2", "M1", 1),
            ]
            return SimpleNamespace(
                status=SimpleNamespace(name="OPTIMAL"),
                best=SimpleNamespace(tasks=tasks),
                objective=4,
                lower_bound=4,
                runtime=0.01,
            )

    monkeypatch.setitem(sys.modules, "pyjobshop", SimpleNamespace(Model=Model))
    monkeypatch.setitem(
        sys.modules, "pyjobshop.constants", SimpleNamespace(MAX_VALUE=2**42)
    )
    return tasks


def test_global_mode_array_is_mapped_to_semantic_ids(fake_backend):
    factory, workload = flexible_case()
    solution = PyJobShopAdapter().solve(
        SolveRequest(factory, workload, "makespan", 60, 42)
    )
    assert solution.schedule == FAST
    assert replay_schedule(factory, workload, solution.schedule).makespan == 4


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_adapter_rejects_wrong_operation_mode_mapping(fake_backend, index):
    factory, workload = flexible_case()
    fake_backend[0].mode = index
    with pytest.raises(ValueError, match="mode mapping"):
        PyJobShopAdapter().solve(SolveRequest(factory, workload, "makespan", 60, 42))


def test_horizon_uses_longest_mode_not_first(fake_backend):
    factory, workload = flexible_case()
    a = workload.orders[0].jobs[0]
    changed = replace(
        a.operations[0],
        modes=(a.operations[0].modes[1], ProcessingMode("long", "M1", 2**42)),
    )
    workload = replace(
        workload,
        orders=(
            replace(
                workload.orders[0],
                jobs=(
                    replace(a, operations=(changed, a.operations[1])),
                    workload.orders[0].jobs[1],
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="horizon"):
        PyJobShopAdapter().solve(SolveRequest(factory, workload, "makespan", 60, 42))


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda e: replace(e, processing_mode_id="unknown"), "mode"),
        (lambda e: replace(e, processing_mode_id="slow"), "machine"),
        (lambda e: replace(e, completion_time=2), "duration"),
    ],
)
def test_bad_multimode_solver_schedule_is_saved_before_failure(
    bundle, monkeypatch, mutation, reason
):
    resolved = resolve_run(run_path(bundle, "mk01_cp"))
    factory, workload = flexible_case()
    resolved = replace(resolved, factory=factory, workload=workload)
    solution = ScheduleSolution(
        (mutation(FAST[0]), *FAST[1:]), SolverStatus.OPTIMAL, 4, 4, 0.01
    )
    monkeypatch.setattr(PyJobShopAdapter, "solve", lambda self, request: solution)
    with pytest.raises(RunFailedError, match=reason) as error:
        run_one(resolved)
    assert (
        json_file(error.value.run_dir, "solver_result.json")["schedule"][0][
            "processing_mode_id"
        ]
        == solution.schedule[0].processing_mode_id
    )
    assert json_file(error.value.run_dir, "summary.json")["makespan"] is None
    assert (
        json_file(error.value.run_dir, "failure.json")["stage"] == "schedule_validation"
    )
