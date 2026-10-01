"""Engineering checks for the scalable rule controller on bundled layouts."""

import json
from pathlib import Path

import pytest

from smartsom.algorithms.production_composition import BoundaryCoordinator
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.production import (
    ScenarioFile,
    WorkloadFile,
    materialize,
    validate_production_scenario,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.studio.templates import load_template_file


def test_clearance_controller_compiles_from_four_author_files():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/test/runs/all_rules_clearance_v4.yaml"
    )
    prepared = compile_experiment(path).entries[0].prepared
    policies = json.loads(prepared.policies_json)
    assert policies["mover"]["implementation"]["name"] == "clearance_shortest_path"
    assert policies["dispatcher"]["implementation"]["parameters"] == {
        "fleet_admission": "traffic",
        "work_in_progress_first": True,
    }
    assert len(json.loads(prepared.evaluation_json)) == 5
    assert (
        json.loads(prepared.composition_json)["matching"]["name"] == "priority_greedy"
    )


@pytest.mark.parametrize("number", (7, 8, 9))
def test_rule_controller_completes_grid_scale_templates(number):
    factory = load_template_file(number).factory
    workload = WorkloadFile.model_validate(
        {
            "schema": "smartsom.workload/v2",
            "profile": {
                "jobs": 4,
                "route": (
                    "operation_1",
                    "operation_2",
                    "operation_3",
                    "operation_4",
                ),
                "nominal_min": 1,
                "nominal_max": 1,
                "due_at": 1000,
                "input_id": "buffer_001",
            },
        }
    )
    settings = ScenarioFile.model_validate(
        {
            "schema": "smartsom.scenario/v2",
            "factory": "unused",
            "workload": "unused",
            "mode": "finite",
            "tick_limit": 1024,
        }
    )
    scenario = materialize(factory, workload, settings, 202)
    validate_production_scenario(scenario)
    simulator = ProductionSimulator(scenario, contract="v3")
    policies = {
        "machine": RulePolicy("machine", "normal_first", seed=11),
        "buffer": RulePolicy("buffer", "edd", seed=12),
        "dispatcher": RulePolicy(
            "dispatcher",
            "nearest",
            seed=13,
            parameters={
                "fleet_admission": "traffic",
                "work_in_progress_first": True,
            },
        ),
        "mover": RulePolicy("mover", "clearance_shortest_path", seed=14),
    }
    bindings = {
        role: {"default": role, "overrides": {}}
        for role in ("machine", "buffer", "dispatcher", "mover")
    }
    coordinator = BoundaryCoordinator(
        simulator, policies, bindings, matching="priority_greedy"
    )
    previous_moves = {}
    reverse_runs = {}
    persistent_cycles = []
    agv_rejections = []
    while not simulator.done:
        row = coordinator.tick()
        agv_rejections.extend(row["rejections"])
        for event in row["events"]:
            if event["kind"] != "move":
                continue
            owner = event["agv"]
            state = row["boundary_state"]["agvs"][owner]
            target = state["target"]
            task = (
                (
                    target["owner"],
                    target["port"],
                    state.get("job"),
                )
                if target
                else None
            )
            previous = previous_moves.get(owner)
            if (
                previous is not None
                and previous[0] == task
                and previous[1] == event["after"]
                and previous[2] == event["before"]
            ):
                reverse_runs[owner] = reverse_runs.get(owner, 0) + 1
                if reverse_runs[owner] >= 2:
                    persistent_cycles.append(
                        (owner, row["tick"], event["before"], event["after"])
                    )
            else:
                reverse_runs[owner] = 0
            previous_moves[owner] = (task, event["before"], event["after"])

    assert simulator.status == "completed"
    assert len(simulator.completed) == 4
    assert not agv_rejections
    assert not persistent_cycles
