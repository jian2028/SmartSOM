import builtins
import hashlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path
from test_static_engine import assert_schedule_is_legal, competition_case

from smartsom.algorithms import SPTPolicy
from smartsom.algorithms.pyjobshop import PyJobShopAdapter
from smartsom.algorithms.solver import ScheduleSolution, SolveRequest, SolverStatus
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest, primitive
from smartsom.dispatch import WaitUntil
from smartsom.domain import ScheduledOperation
from smartsom.engine import ReplayError, Simulator, replay, replay_schedule
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/ft06"


@pytest.fixture
def cp_case(bundle):
    edit(
        run_path(bundle),
        lambda data: data.update(algorithm="../algorithms/cp_sat.yaml"),
    )
    edit(
        bundle / "configs/scenarios/competition.yaml",
        lambda data: data.update(visibility="full_static"),
    )
    return resolve_run(run_path(bundle))


def competition_solution(status=SolverStatus.OPTIMAL, bound=6):
    factory, workload, actions = competition_case()
    schedule = replay(factory, workload, actions).schedule
    return ScheduleSolution(schedule, status, 6, bound, 0.01)


def test_cp_default_budget_information_and_effective_seed(cp_case):
    assert cp_case.run.budget.solver_time_limit_seconds == 60
    assert cp_case.algorithm.algorithm.required_information == "full_static"
    assert [(seed.domain, seed.value) for seed in cp_case.seeds if seed.consumed] == [
        ("solver", 9725273414194853204)
    ]
    request = SolveRequest(
        cp_case.factory, cp_case.workload, "makespan", 60, 9725273414194853204
    )
    assert request.backend_seed == 9725273414194853204 % 2**31
    with pytest.raises(FrozenInstanceError):
        request.solver_seed = 1
    with pytest.raises(FrozenInstanceError):
        competition_solution().schedule = ()


@pytest.mark.parametrize(
    "status,bound,optimal",
    [(SolverStatus.OPTIMAL, 6, True), (SolverStatus.FEASIBLE, 5, False)],
)
def test_solver_result_is_saved_before_exact_replay(
    cp_case, monkeypatch, status, bound, optimal
):
    solution = competition_solution(status, bound)

    def solve(self, request):
        assert request.factory is cp_case.factory
        assert request.workload is cp_case.workload
        assert request.solver_time_limit_seconds == 60
        return solution

    monkeypatch.setattr(PyJobShopAdapter, "solve", solve)
    original = ScheduleReplayPolicy.select_action

    def check_saved(self, context):
        dirs = list(Path(cp_case.run.output_root).iterdir())
        assert len(dirs) == 1
        assert json_file(dirs[0], "solver_result.json")["schedule"] == primitive(
            solution.schedule
        )
        return original(self, context)

    monkeypatch.setattr(ScheduleReplayPolicy, "select_action", check_saved)
    run = run_one(cp_case)
    assert run.simulation_result == replay_schedule(
        cp_case.factory, cp_case.workload, solution.schedule
    )
    summary = json_file(run.run_dir, "summary.json")
    assert summary["solver_status"] == status
    assert summary["proven_optimal"] is optimal
    assert summary["makespan"] == 6
    evidence = json_file(run.run_dir, "solver_result.json")
    assert evidence["gap"] == (6 - bound) / 6
    assert evidence["num_workers"] == 1
    assert evidence["backend_seed"] == evidence["solver_seed"] % 2**31
    manifest = json_file(run.run_dir, "manifest.json")
    assert manifest["information_projection"] == "full_static"
    assert manifest["interface_kind"] == "offline_solver"
    assert manifest["provider_implementation"].endswith("PyJobShopAdapter")
    for name, sha in manifest["artifacts"].items():
        assert hashlib.sha256((run.run_dir / name).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize(
    "status", [SolverStatus.INFEASIBLE, SolverStatus.TIME_LIMIT, SolverStatus.UNKNOWN]
)
def test_no_incumbent_retains_strict_json_without_success_metrics(
    cp_case, monkeypatch, status
):
    solution = ScheduleSolution((), status, float("inf"), float("nan"), 60)
    monkeypatch.setattr(PyJobShopAdapter, "solve", lambda self, request: solution)
    with pytest.raises(RunFailedError, match="no feasible incumbent") as error:
        run_one(cp_case)
    directory = error.value.run_dir
    result = json_file(directory, "solver_result.json")
    assert result["objective"] is result["bound"] is result["gap"] is None
    assert result["status"] == status
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_file(directory, "failure.json")["stage"] == "schedule_validation"
    assert json_lines(directory, "trace.jsonl") == []
    assert json_lines(directory, "metrics.jsonl") == []


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda s: replace(s, schedule=s.schedule[1:]), "missing"),
        (lambda s: replace(s, objective=7), "objective"),
        (lambda s: replace(s, bound=7), "bound"),
        (lambda s: replace(s, bound=5), "OPTIMAL"),
        (lambda s: replace(s, schedule=(*s.schedule, s.schedule[0])), "duplicate"),
    ],
)
def test_inconsistent_solver_output_is_saved_and_rejected(
    cp_case, monkeypatch, mutation, reason
):
    solution = mutation(competition_solution())
    monkeypatch.setattr(PyJobShopAdapter, "solve", lambda self, request: solution)
    with pytest.raises(RunFailedError, match=reason) as error:
        run_one(cp_case)
    assert (error.value.run_dir / "solver_result.json").exists()
    assert json_file(error.value.run_dir, "summary.json")["makespan"] is None


def test_replay_failure_keeps_solver_output_and_actual_trace(cp_case, monkeypatch):
    monkeypatch.setattr(
        PyJobShopAdapter, "solve", lambda self, request: competition_solution()
    )

    def fail(self, result):
        raise ReplayError("actual schedule differs from solver output")

    monkeypatch.setattr(ScheduleReplayPolicy, "verify_result", fail)
    with pytest.raises(RunFailedError, match="differs") as error:
        run_one(cp_case)
    directory = error.value.run_dir
    assert json_file(directory, "failure.json")["stage"] == "schedule_verification"
    assert json_file(directory, "summary.json")["makespan"] is None
    assert json_file(directory, "solver_result.json")["objective"] == 6
    assert json_lines(directory, "trace.jsonl")[-1]["kind"] == "terminate"
    assert all(
        row["kind"] != "terminal" for row in json_lines(directory, "metrics.jsonl")
    )


def test_solver_exception_and_cli_failure_are_recorded(bundle, cp_case, monkeypatch):
    def fail(self, request):
        raise RuntimeError("model construction failed")

    monkeypatch.setattr(PyJobShopAdapter, "solve", fail)
    assert main(["run", str(run_path(bundle))]) == 1
    directory = next(Path(cp_case.run.output_root).iterdir())
    assert json_file(directory, "failure.json")["stage"] == "solving"
    assert not (directory / "solver_result.json").exists()
    assert json_file(directory, "summary.json")["makespan"] is None


def test_missing_extra_is_explicit_and_never_imported_by_validate(
    bundle, cp_case, monkeypatch
):
    original = builtins.__import__

    def block(name, *args, **kwargs):
        if name.split(".")[0] in {"pyjobshop", "ortools"}:
            raise ImportError("blocked optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    assert main(["validate", str(run_path(bundle))]) == 0
    assert not Path(cp_case.run.output_root).exists()
    with pytest.raises(RunFailedError, match="optional cp extra") as error:
        run_one(cp_case)
    assert json_file(error.value.run_dir, "summary.json")["makespan"] is None


@pytest.mark.parametrize("value", [0, -1, True, "60", float("inf"), float("nan")])
def test_invalid_solver_budget_fails_resolution(bundle, value):
    edit(
        run_path(bundle, "ft06_cp"),
        lambda data: data.update(budget={"solver_time_limit_seconds": value}),
    )
    assert main(["validate", str(run_path(bundle, "ft06_cp"))]) == 2
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "relative,mutation",
    [
        (
            "configs/scenarios/ft06.yaml",
            lambda d: d.update(visibility="decision_context"),
        ),
        (
            "configs/algorithms/cp_sat.yaml",
            lambda d: d["algorithm"].pop("required_information"),
        ),
        (
            "configs/algorithms/cp_sat.yaml",
            lambda d: d["algorithm"].update(interface_kind="online_policy"),
        ),
        (
            "configs/algorithms/cp_sat.yaml",
            lambda d: d["algorithm"].update(parameters={"num_workers": 4}),
        ),
        ("configs/algorithms/cp_sat.yaml", lambda d: d["algorithm"].update(seed=1)),
        ("configs/runs/ft06_cp.yaml", lambda d: d["budget"].update(extra=1)),
    ],
)
def test_unsupported_cp_configuration_fails_before_allocation(
    bundle, relative, mutation
):
    edit(bundle / relative, mutation)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "ft06_cp"))
    assert not (bundle / "runs").exists()


def test_online_budget_rejected_and_script_wait_accepted(bundle):
    edit(
        run_path(bundle),
        lambda data: data.update(budget={"solver_time_limit_seconds": 1}),
    )
    with pytest.raises(ConfigurationError, match="online providers"):
        resolve_run(run_path(bundle))
    edit(run_path(bundle), lambda data: data.pop("budget"))
    edit(
        bundle / "configs/algorithms/competition_script.yaml",
        lambda data: data["algorithm"]["parameters"]["actions"].insert(0, {"until": 5}),
    )
    run = run_one(resolve_run(run_path(bundle)))
    assert run.simulation_result.actions[0] == WaitUntil(5)
    assert run.simulation_result.makespan == 11


def test_ft06_reference_has_independent_source_and_interval_checks():
    resolved = resolve_run(ROOT / "configs/runs/ft06_spt.yaml")
    reference = json_file(REFERENCE, "schedule.json")
    schedule = tuple(ScheduledOperation(**entry) for entry in reference["schedule"])
    source = json_file(REFERENCE, "sources.json")
    for entry in source["sources"]:
        assert (
            hashlib.sha256((REFERENCE / entry["snapshot"]).read_bytes()).hexdigest()
            == entry["snapshot_sha256"]
        )
    assert source["workload_sha256"] == resolved.workload_sha256
    assert source["schedule_sha256"] == digest(schedule)
    lib = json_file(REFERENCE, "jobshoplib_ft06.json")
    lines = [
        line
        for line in (REFERENCE / "instance.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    assert lines[0].split() == ["6", "6"]
    routes = [list(zip(*[iter(map(int, line.split()))] * 2)) for line in lines[1:]]
    starts = [
        list(map(int, line.split()))
        for line in (REFERENCE / "jobshoplab_start_times.txt").read_text().splitlines()
    ]
    indexed = {entry.operation_id: entry for entry in schedule}
    for j, job in enumerate(resolved.workload.orders[0].jobs):
        for i, op in enumerate(job.operations):
            machine, duration = routes[j][i]
            assert (machine, duration) == (
                lib["machines_matrix"][j][i],
                lib["duration_matrix"][j][i],
            )
            assert op.modes[0].machine_id == f"M{machine}"
            assert op.modes[0].nominal_ticks == duration
            assert indexed[op.operation_id].start_time == starts[j][machine]
            assert (
                indexed[op.operation_id].completion_time
                == starts[j][machine] + duration
            )
    result = replay_schedule(resolved.factory, resolved.workload, reversed(schedule))
    assert result.schedule == schedule
    assert result.makespan == reference["makespan"] == 55
    assert sum(record.kind == "wait" for record in result.trace) > 0
    assert replay(resolved.factory, resolved.workload, result.actions) == result
    assert_schedule_is_legal(resolved.workload, result)
    bounds = json_file(REFERENCE, "scheduleopt_bounds.json")["entry"]
    assert (
        bounds["ub"]["value"]
        == bounds["lb"]["value"]
        == lib["metadata"]["optimum"]
        == 55
    )
    assert bounds["ub"]["certificate"] == "no"


def test_ft06_spt_uses_same_runner_and_has_a_complete_legal_schedule(bundle):
    resolved = resolve_run(run_path(bundle, "ft06_spt"))
    run = run_one(resolved)
    assert run.simulation_result == Simulator(resolved.factory, resolved.workload).run(
        SPTPolicy()
    )
    assert_schedule_is_legal(resolved.workload, run.simulation_result)
    assert all(
        not isinstance(action, WaitUntil) for action in run.simulation_result.actions
    )
    assert (
        json_file(run.run_dir, "manifest.json")["information_projection"]
        == "decision_context"
    )


def test_reference_replay_is_independent_of_hash_seed_and_cwd(tmp_path):
    script = """
import json
from pathlib import Path
from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.domain import ScheduledOperation
from smartsom.engine import replay_schedule
root = Path(__import__('sys').argv[1])
r = resolve_run(root/'configs/runs/ft06_spt.yaml')
schedule = [ScheduledOperation(**x) for x in json.loads((root/'data/reference/ft06/schedule.json').read_text())['schedule']]
print(canonical_json(replay_schedule(r.factory,r.workload,schedule)))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script, str(ROOT)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "123")
    ]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("solver_seed", True),
        ("solver_seed", -1),
        ("solver_seed", 2**64),
        ("solver_time_limit_seconds", True),
        ("solver_time_limit_seconds", 0),
        ("objective", "tardiness"),
    ],
)
def test_solver_request_rejects_unsupported_values(field, value):
    factory, workload, _ = competition_case()
    values = dict(
        factory=factory,
        workload=workload,
        objective="makespan",
        solver_time_limit_seconds=60,
        solver_seed=1,
    )
    values[field] = value
    with pytest.raises(ValueError):
        SolveRequest(**values)
