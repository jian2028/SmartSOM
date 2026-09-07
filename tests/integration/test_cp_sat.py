"""The CP CI job requires this suite; the base environment may exclude it."""

import importlib.metadata
import importlib.util
import json
import os
import runpy
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


@pytest.mark.parametrize(
    "name,optimum,operations",
    [("ft06", 55, 36), ("pyjobshop_fjsp", 6, 9), ("mk01", 40, 55)],
)
def test_real_optimum_and_exact_replay_through_runner(
    tmp_path, name, optimum, operations
):
    resolved = resolve_run(ROOT / f"configs/runs/{name}_cp.yaml")
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
        == optimum
    )
    assert len(solution["schedule"]) == operations
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


def test_real_adapter_retains_same_machine_modes_and_maps_global_indices():
    from smartsom.domain import (
        FactorySpec,
        Job,
        Machine,
        Operation,
        Order,
        ProcessingMode,
        WorkloadInstance,
    )

    factory = FactorySpec((Machine("M1"),))
    modes = (
        ProcessingMode("slow", "M1", 5),
        ProcessingMode("a", "M1", 2),
        ProcessingMode("z", "M1", 2),
    )
    workload = WorkloadInstance(
        (
            Order(
                "order",
                (
                    Job(
                        "job",
                        (
                            Operation(
                                "second",
                                (ProcessingMode("finish", "M1", 1),),
                                ("first",),
                            ),
                            Operation("first", modes),
                        ),
                    ),
                ),
            ),
        )
    )
    solution = PyJobShopAdapter().solve(
        SolveRequest(factory, workload, "makespan", 60, 42)
    )
    solution.require_incumbent()
    assert solution.objective == solution.bound == 3
    assert solution.schedule[0].processing_mode_id in ("a", "z")
    assert solution.schedule[1].processing_mode_id == "finish"
    assert replay_schedule(factory, workload, solution.schedule).makespan == 3


@pytest.mark.parametrize(
    "name,optimum", [("ft06", 55), ("pyjobshop_fjsp", 6), ("mk01", 40)]
)
def test_real_adapter_uses_explicit_predecessors_and_semantic_mapping(name, optimum):
    resolved = resolve_run(ROOT / f"configs/runs/{name}_cp.yaml")
    factory = replace(
        resolved.factory, machines=tuple(reversed(resolved.factory.machines))
    )
    workload = replace(
        resolved.workload,
        orders=tuple(
            replace(
                order,
                jobs=tuple(
                    replace(
                        job,
                        operations=tuple(
                            replace(op, modes=tuple(reversed(op.modes)))
                            for op in reversed(job.operations)
                        ),
                    )
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
    assert solution.objective == solution.bound == optimum
    result = replay_schedule(factory, workload, solution.schedule)
    assert result.schedule == solution.schedule
    assert result.makespan == optimum


@pytest.mark.parametrize("index", range(4))
def test_fixed_breaks_external_reference_and_core_replay(index):
    reference = runpy.run_path(str(ROOT / "scripts/validate_machine_events.py"))
    case = json.loads((ROOT / "data/reference/machine_events/cases.json").read_text())[
        "cases"
    ][index]
    result = reference["core_reference"](case)
    external = reference["pyjobshop_reference"](case)
    assert external["objective"] == external["bound"] == result.makespan
