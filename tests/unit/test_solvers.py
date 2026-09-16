"""Grid solver rejection and independently preserved historical scheduling evidence."""

import builtins
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from reference_cases import competition_case
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.algorithms.solver import SolveRequest
from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import digest
from smartsom.domain import ScheduledOperation
from smartsom.experiments import run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/ft06"


@pytest.mark.parametrize(
    "scenario",
    ["production_hand", "crossing", "ft06", "pyjobshop_fjsp", "quality_fixed"],
)
@pytest.mark.parametrize("command", ["validate", "run"])
def test_cp_cannot_silently_execute_the_matrix_core(
    bundle, monkeypatch, scenario, command, capsys
):
    path = run_path(bundle, "ft06_cp")
    edit(path, lambda data: data.update(scenario=f"../scenarios/{scenario}.yaml"))
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"pyjobshop", "ortools"}:
            pytest.fail("unsupported grid solver imported an optional backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(
        "smartsom.engine.production.ProductionSimulator",
        lambda *a, **k: pytest.fail("unsupported solver started simulation"),
    )
    assert main([command, str(path)]) == 2
    assert "no grid production adapter" in capsys.readouterr().err
    assert not (bundle / "runs").exists()


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


def test_online_solver_budget_is_rejected_before_allocation(bundle):
    path = run_path(bundle, "production_hand")
    edit(path, lambda data: data.update(budget={"solver_time_limit_seconds": 1}))
    with pytest.raises(ConfigurationError, match="solver|online"):
        resolve_run(path)
    assert not (bundle / "runs").exists()


def test_ft06_reference_sources_and_hand_checked_intervals_remain_historical():
    scenario = resolve_run(ROOT / "configs/runs/ft06_spt.yaml").resolved.scenario
    reference = json_file(REFERENCE, "schedule.json")
    schedule = tuple(ScheduledOperation(**entry) for entry in reference["schedule"])
    source = json_file(REFERENCE, "sources.json")
    for entry in source["sources"]:
        assert (
            hashlib.sha256((REFERENCE / entry["snapshot"]).read_bytes()).hexdigest()
            == entry["snapshot_sha256"]
        )
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
    assert len(indexed) == 36
    for j, demand in enumerate(scenario.demands):
        previous = 0
        for i, step in enumerate(demand.steps):
            machine, duration = routes[j][i]
            assert (machine, duration) == (
                lib["machines_matrix"][j][i],
                lib["duration_matrix"][j][i],
            )
            assert dict(step.machine_nominal_ticks) == {f"M{machine}": duration}
            entry = indexed[step.operation_id]
            assert entry.start_time == starts[j][machine]
            assert entry.completion_time == entry.start_time + duration
            assert entry.start_time >= previous
            previous = entry.completion_time
    for machine in {entry.machine_id for entry in schedule}:
        intervals = sorted(
            (entry.start_time, entry.completion_time)
            for entry in schedule
            if entry.machine_id == machine
        )
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    assert (
        max(entry.completion_time for entry in schedule) == reference["makespan"] == 55
    )
    bounds = json_file(REFERENCE, "scheduleopt_bounds.json")["entry"]
    assert (
        bounds["ub"]["value"]
        == bounds["lb"]["value"]
        == lib["metadata"]["optimum"]
        == 55
    )
    assert bounds["ub"]["certificate"] == "no"
    # 55 is a processing-only reference, never a grid transport optimum claim.


def test_ft06_grid_spt_uses_shared_runner_and_completes_all_operations(bundle):
    prepared = resolve_run(run_path(bundle, "ft06_spt"))
    run = run_one(prepared, verbose=False)
    assert run.simulation_result.status == "completed"
    assert audit(run.run_dir)["status"] == "passed"
    events = [
        e for row in json_lines(run.run_dir, "trace.jsonl") for e in row["events"]
    ]
    completions = [e for e in events if e["kind"] == "processing_completed"]
    assert {e["operation"] for e in completions} == {
        step.operation_id
        for d in prepared.resolved.scenario.demands
        for step in d.steps
    }
    assert len(completions) == 36
    assert run.simulation_result.makespan > 55


def test_grid_trace_is_independent_of_hash_seed_and_cwd(tmp_path):
    code = """from smartsom.config import resolve_run
from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.engine.production import ProductionSimulator
from smartsom.config.codec import canonical_json
from pathlib import Path
import sys
case=resolve_run(Path(sys.argv[1])/'configs/runs/ft06_spt.yaml').resolved.scenario
sim=ProductionSimulator(case); policy=GreedyProductionPolicy(case.factory,rule='spt'); rows=[]
while not sim.done and sim.tick<1024:
 view=sim.decision(rankings=policy.rank(sim.decision()))
 rows.append(sim.step(policy.act(view)))
print(canonical_json(rows))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(ROOT)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": seed},
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
