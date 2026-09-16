"""One authoring format, frozen inputs and independent environmental streams."""

import json
from pathlib import Path

import pytest
import yaml

from smartsom.api import load_config, prepare, show_config
from smartsom.config.codec import primitive
from smartsom.config.production import scenario_from_snapshot
from smartsom.engine.production import ProductionSimulator


def test_generated_world_is_repeatable_and_round_trips_without_sources():
    config = load_config("configs/runs/dynamic_production.yaml")
    first = prepare(config, training=False).resolved.scenario
    assert prepare(config, training=False).resolved.scenario == first
    restored = scenario_from_snapshot(primitive(first))
    assert restored == first
    assert (
        ProductionSimulator(restored).snapshot()
        == ProductionSimulator(first).snapshot()
    )
    assert first.outages
    changed = prepare(
        config.model_copy(update={"seed": 101}), training=False
    ).resolved.scenario
    assert (changed.outages, changed.demands) != (first.outages, first.demands)


def test_presentation_controls_do_not_change_science():
    config = load_config("configs/runs/production_hand.yaml")
    before = show_config(config)
    config.logging.verbose = False
    config.evaluation.record = False
    config.output.root = "/tmp/elsewhere"
    assert show_config(config)["scientific_sha256"] == before["scientific_sha256"]


def test_duplicate_yaml_mapping_is_an_error(tmp_path):
    source = tmp_path / "run.yaml"
    source.write_text("schema: smartsom.run/v2\nscenario: a.yaml\nscenario: b.yaml\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_config(source)


def test_unknown_schema_and_unknown_fields_do_not_fall_back(tmp_path):
    source = tmp_path / "run.yaml"
    for data in (
        {"schema": "smartsom.run/v1", "scenario": "x"},
        {"schema": "smartsom.run/v2", "scenario": "x", "typo": 1},
    ):
        source.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError):
            load_config(source)


def test_relocated_recipe_keeps_frozen_inputs(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "configs", tmp_path / "configs")
    a = show_config(load_config(root / "configs/runs/production_hand.yaml"))
    b = show_config(load_config(tmp_path / "configs/runs/production_hand.yaml"))
    assert a["scenario"] == b["scenario"]
    assert a["scientific_sha256"] == b["scientific_sha256"]
    shutil.rmtree(tmp_path / "configs")
    assert (
        scenario_from_snapshot(b["scenario"])
        == prepare(
            load_config(root / "configs/runs/production_hand.yaml"), training=False
        ).resolved.scenario
    )


def test_announcements_survive_json_wire_round_trip():
    from dataclasses import replace

    case = prepare(
        load_config("configs/runs/dynamic_production.yaml"), training=False
    ).resolved.scenario
    case = replace(case, demands=(replace(case.demands[0], release_at=3, reveal_at=0),))
    state = ProductionSimulator(case).snapshot()
    assert json.loads(json.dumps(state)) == state


def test_sampled_routes_and_machine_times_keep_generation_controls():
    from smartsom.config.production import ScenarioFile, WorkloadFile, materialize

    factory = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved.scenario.factory
    workload = WorkloadFile.model_validate_json(
        json.dumps(
            {
                "schema": "smartsom.workload/v2",
                "profile": {
                    "jobs": 12,
                    "operation_types": ["drill"],
                    "min_operations": 2,
                    "max_operations": 4,
                    "nominal_min": 3,
                    "nominal_max": 7,
                    "machine_duration_variation": True,
                },
            }
        )
    )
    settings = ScenarioFile(
        schema="smartsom.scenario/v2", factory="unused", workload="unused"
    )
    first = materialize(factory, workload, settings, 42)
    assert first == materialize(factory, workload, settings, 42)
    assert first.demands != materialize(factory, workload, settings, 43).demands
    assert {len(d.steps) for d in first.demands} == {2, 3, 4}
    for demand in first.demands:
        for step in demand.steps:
            assert 3 <= step.nominal_ticks <= 7
            assert dict(step.machine_nominal_ticks).keys() == {"machine"}
            assert 3 <= dict(step.machine_nominal_ticks)["machine"] <= 7
    # Invalid catalogs must fail even if there are zero jobs to materialize.
    bad = workload.model_copy(deep=True)
    bad.profile.operation_types = ("unknown",)
    bad.profile.jobs = 0
    with pytest.raises(ValueError, match="unknown operation type"):
        materialize(factory, bad, settings, 42)


def test_outage_generation_stops_at_its_authored_window():
    from smartsom.config.production import ScenarioFile, WorkloadFile, materialize

    case = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved.scenario
    settings = ScenarioFile.model_validate_json(
        json.dumps(
            {
                "schema": "smartsom.scenario/v2",
                "factory": "unused",
                "workload": "unused",
                "tick_limit": 1000,
                "outage_profiles": [
                    {
                        "machine_id": "machine",
                        "mean_uptime_ticks": 0.01,
                        "repair_min": 1,
                        "repair_max": 1,
                        "until_tick": 5,
                    }
                ],
            }
        )
    )
    result = materialize(
        case.factory,
        WorkloadFile(schema="smartsom.workload/v2", demands=case.demands),
        settings,
        42,
    )
    assert [(o.start, o.end) for o in result.outages] == [(1, 2), (3, 4)]


def test_scripted_grid_commands_and_rule_budget(tmp_path):
    from smartsom.algorithms.production import GreedyProductionPolicy
    from smartsom.config.production import compile_algorithm
    from smartsom.experiments.runner import RunFailedError, run_one

    config = load_config("configs/runs/production_hand.yaml")
    config.output.root = str(tmp_path / "runs")
    config.logging.verbose = False
    case = prepare(config, training=False).resolved.scenario
    sim = ProductionSimulator(case)
    policy = GreedyProductionPolicy(case.factory)
    commands = []
    while not sim.done:
        view = sim.decision(policy.rank(sim.decision()))
        command = policy.act(view)
        commands.append(primitive(command))
        sim.step(command)
    source = tmp_path / "script.yaml"
    document = {
        "schema": "smartsom.algorithm/v1",
        "algorithm": {
            "provider": "builtin.scripted",
            "parameters": {"commands": commands},
        },
    }
    source.write_text(yaml.safe_dump(document))
    config.algorithm.source = str(source)
    compiled = compile_algorithm(config, training=False)
    assert len(compiled.commands) == 8
    result = run_one(prepare(config, training=False))
    assert result.simulation_result.status == "completed"
    assert result.simulation_result.final_state == sim.snapshot()
    config.training.max_ticks = 4
    limited = run_one(prepare(config, training=False))
    assert limited.simulation_result.status == "truncated"
    assert limited.simulation_result.final_state["tick"] == 4
    assert (
        json.loads((limited.run_dir / "run.json").read_text())["reason"]
        == "budget_exhausted"
    )
    config.training.max_ticks = 100
    document["algorithm"]["parameters"]["commands"] = commands[:2]
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(RunFailedError, match="commands exhausted") as error:
        run_one(prepare(config, training=False))
    assert (
        json.loads((error.value.run_dir / "run.json").read_text())["status"] == "failed"
    )
    document["algorithm"]["parameters"] = {
        "actions": [{"operation_id": "op", "processing_mode_id": "standard"}]
    }
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(
        ValueError, match="historical scripted actions require migration"
    ):
        compile_algorithm(config, training=False)
