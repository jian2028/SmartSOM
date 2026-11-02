"""Batch02 small grid: original rules stall, traffic-tolerant variants finish."""

import json
from pathlib import Path

import yaml

from smartsom.algorithms.production_composition import BoundaryCoordinator
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.production import scenario_from_snapshot
from smartsom.engine.production import ProductionSimulator

ROOT = Path(__file__).resolve().parents[2]


def grid_case(tmp_path):
    algorithm = yaml.safe_load(
        (ROOT / "configs/algorithms/batch02_all_rules.yaml").read_text()
    )
    algorithm["agents"]["mover"]["default"]["name"] = "shortest_path"
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    experiment = {
        "schema": "smartsom.experiment-config/v4",
        "task": "evaluate",
        "factory": str(ROOT / "configs/factories/batch02_small_h1.yaml"),
        "workload": str(ROOT / "configs/workloads/batch02_small_low.yaml"),
        "algorithm": "algorithm.yaml",
        "data_seed": 909,
        "runtime": {"environment": {"tick_limit": 8192, "transport": {"mode": "grid"}}},
        "evaluation": {"replications": 1},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    plan = compile_experiment(tmp_path / "experiment.yaml")
    case = json.loads(plan.entries[0].prepared.evaluation_json)[0]
    return scenario_from_snapshot(case["scenario"])


def drive(scenario, dispatcher, mover, *, stall=1500):
    sim = ProductionSimulator(scenario, contract="v3")
    names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": dispatcher,
        "mover": mover,
    }
    driver = BoundaryCoordinator(
        sim,
        {role: RulePolicy(role, name, seed=7) for role, name in names.items()},
        {role: {"default": role} for role in names},
        "global_optimal",
    )
    shipped, last = 0, 0
    while not sim.done and sim.tick - last < stall:
        driver.tick()
        if len(sim.completed) != shipped:
            shipped, last = len(sim.completed), sim.tick
    return sim


def test_traffic_tolerant_rules_finish_where_original_rules_stall(tmp_path):
    scenario = grid_case(tmp_path)
    stalled = drive(scenario, "nearest", "shortest_path")
    assert not stalled.done and len(stalled.completed) < len(stalled.demands)
    finished = drive(scenario, "nearest_sticky", "shortest_path_yield")
    assert finished.status == "completed"
    assert len(finished.completed) == len(finished.demands) == 64
