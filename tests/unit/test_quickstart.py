"""The introductory workload stays editable, reproducible and reusable as JSON."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from smartsom.config import ConfigurationError, load_resolved_run, resolve_run
from smartsom.config.experiment import PRESET_ROOT, load_preset, prepare
from smartsom.experiments import run_one
from smartsom.experiments.cli import main

ROOT = Path(__file__).resolve().parents[2]
INPUTS = (
    "configs/factories/factory_test.yaml",
    "configs/workloads/workload_test.yaml",
    "configs/scenarios/scenario_test.yaml",
    "configs/algorithms/algorithm_test.yaml",
    "configs/runs/run_test.yaml",
)


@pytest.fixture
def project(tmp_path):
    for name in INPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_packaged_test_preset_matches_repository_inputs():
    for name in INPUTS:
        assert (PRESET_ROOT / name).read_bytes() == (ROOT / name).read_bytes()
    configured = resolve_run(ROOT / "configs/runs/run_test.yaml").resolved
    bundled = prepare(load_preset("test"), training=False).resolved
    assert bundled.scenario == configured.scenario
    assert bundled.algorithm == configured.algorithm
    assert len(configured.scenario.factory.machines) == 2
    assert len(configured.scenario.demands) == 2
    assert sum(len(d.steps) for d in configured.scenario.demands) == 4
    assert configured.scenario.seed == 42
    assert configured.algorithm.provider == "builtin.spt"
    with pytest.raises(ConfigurationError, match="unknown preset"):
        load_preset("competition")


def test_generated_workload_scales_and_remains_reproducible(project):
    path = project / "configs/runs/run_test.yaml"
    original = resolve_run(path)
    assert original.resolved.workload_json == resolve_run(path).resolved.workload_json
    profile_path = project / "configs/workloads/workload_test.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["profile"]["jobs"] = 7
    profile_path.write_text(yaml.safe_dump(profile))
    larger = resolve_run(path)
    assert len(larger.resolved.scenario.demands) == 7
    assert sum(len(d.steps) for d in larger.resolved.scenario.demands) == 14
    assert larger.resolved.workload_json != original.resolved.workload_json
    assert larger.resolved.workload_json == resolve_run(path).resolved.workload_json
    result = run_one(larger, verbose=False).simulation_result
    assert result.status == "completed" and result.qualified_demands == 7


def test_realized_json_reuses_workload_and_complete_spt_result(project):
    path = project / "configs/runs/run_test.yaml"
    generated = resolve_run(path)
    first = run_one(generated)
    (project / "configs/workloads/workload_test.json").write_text(
        generated.resolved.workload_json
    )
    scenario_path = project / "configs/scenarios/scenario_test.yaml"
    scenario = yaml.safe_load(scenario_path.read_text())
    scenario["workload"] = "../workloads/workload_test.json"
    scenario_path.write_text(yaml.safe_dump(scenario))
    fixed = resolve_run(path)
    assert fixed.resolved.scenario == generated.resolved.scenario
    assert fixed.resolved.workload_json == generated.resolved.workload_json
    assert run_one(fixed, verbose=False).simulation_result == first.simulation_result
    run = yaml.safe_load(path.read_text())
    run["seed"] = 99
    path.write_text(yaml.safe_dump(run))
    assert (
        resolve_run(path).resolved.scenario.demands
        == generated.resolved.scenario.demands
    )
    # The original generation profile also survives independently in run.json.
    assert load_resolved_run(first.run_dir / "run.json") == generated


def test_readme_commands_and_relocated_references(project, monkeypatch, capsys):
    monkeypatch.chdir(project.parent)
    path = str(project / "configs/runs/run_test.yaml")
    assert main(["validate", "--config", path]) == 0
    assert "valid" in capsys.readouterr().out
    assert main(["show-config", "--config", path]) == 0
    capsys.readouterr()
    assert not (project / "runs").exists()
    assert main(["run", "--config", path]) == 0
    assert "completed" in capsys.readouterr().out
    records = list((project / "runs").glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "completed"
    assert len(record["inputs"]["scenario"]["demands"]) == 2
    assert record["result"]["tick"] == 30
    assert (records[0].parent / "trace.jsonl").is_file()


def test_shared_factory_references_resolve_after_rename():
    for path in (ROOT / "configs/scenarios").glob("*.yaml"):
        scenario = yaml.safe_load(path.read_text())
        if scenario.get("factory") == "../factories/factory_test.yaml":
            # Exercise the ordinary run resolver for every affected recipe.
            for run_path in (ROOT / "configs/runs").glob("*.yaml"):
                run = yaml.safe_load(run_path.read_text())
                if run.get("scenario") == f"../scenarios/{path.name}":
                    resolved = resolve_run(run_path)
                    assert len(resolved.resolved.scenario.factory.machines) == 2
