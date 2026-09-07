"""The CP CI job requires this suite; the base environment may exclude it."""

import importlib.metadata
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.algorithms.pyjobshop import PyJobShopAdapter
from smartsom.algorithms.solver import SolveRequest, SolverStatus
from smartsom.config import resolve_run
from smartsom.engine import replay, replay_schedule
from smartsom.experiments import run_one

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.cp


@pytest.fixture(autouse=True)
def cp_environment():
    if importlib.util.find_spec("pyjobshop") is None:
        if os.environ.get("SMARTSOM_REQUIRE_CP") == "1":
            pytest.fail("CP acceptance requires the locked cp extra")
        pytest.skip("optional cp extra is not installed in the base environment")
    assert importlib.metadata.version("pyjobshop") == "0.0.9"
    assert importlib.metadata.version("ortools") == "9.12.4544"


def test_real_ft06_optimum_and_exact_replay_through_runner(tmp_path):
    resolved = resolve_run(ROOT / "configs/runs/ft06_cp.yaml")
    resolved = replace(
        resolved,
        run=resolved.run.model_copy(update={"output_root": str(tmp_path / "runs")}),
    )
    run = run_one(resolved)
    solution = json.loads((run.run_dir / "solver_result.json").read_text())
    assert solution["status"] == "OPTIMAL"
    assert (
        solution["objective"]
        == solution["bound"]
        == run.simulation_result.makespan
        == 55
    )
    assert len(solution["schedule"]) == 36
    assert solution["backend_seed"] == 1017279828
    assert solution["num_workers"] == 1
    assert solution["gap"] == 0
    result = run.simulation_result
    assert (
        replay_schedule(resolved.factory, resolved.workload, result.schedule) == result
    )
    assert replay(resolved.factory, resolved.workload, result.actions) == result
    manifest = json.loads((run.run_dir / "manifest.json").read_text())
    assert manifest["source"]["packages"]["pyjobshop"] == "0.0.9"
    assert manifest["source"]["packages"]["ortools"] == "9.12.4544"


def test_real_adapter_uses_explicit_predecessors_and_semantic_mapping():
    resolved = resolve_run(ROOT / "configs/runs/ft06_cp.yaml")
    factory = replace(
        resolved.factory, machines=tuple(reversed(resolved.factory.machines))
    )
    workload = replace(
        resolved.workload,
        orders=tuple(
            replace(
                order,
                jobs=tuple(
                    replace(job, operations=tuple(reversed(job.operations)))
                    for job in reversed(order.jobs)
                ),
            )
            for order in reversed(resolved.workload.orders)
        ),
    )
    solution = PyJobShopAdapter().solve(
        SolveRequest(factory, workload, "makespan", 60, 42)
    )
    solution.require_incumbent()
    assert solution.status == SolverStatus.OPTIMAL
    assert solution.objective == solution.bound == 55
    result = replay_schedule(factory, workload, solution.schedule)
    assert result.schedule == solution.schedule
    assert result.makespan == 55
