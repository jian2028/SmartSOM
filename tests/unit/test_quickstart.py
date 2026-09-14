"""The introductory workload stays editable, reproducible and reusable as JSON."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from smartsom.config import ConfigurationError, resolve_run
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
    configured = resolve_run(ROOT / "configs/runs/run_test.yaml")
    bundled = prepare(load_preset("test"), training=False).resolved
    assert bundled.factory == configured.factory
    assert bundled.workload == configured.workload
    assert bundled.algorithm == configured.algorithm
    assert bundled.seeds == configured.seeds
    assert len(configured.factory.machines) == 2
    assert len(configured.workload.orders) == 1
    assert len(configured.workload.orders[0].jobs) == 2
    assert len(configured.workload.operations) == 4
    assert configured.run.seed == 42
    assert configured.algorithm.algorithm.provider == "builtin.spt"
    with pytest.raises(ConfigurationError, match="unknown preset"):
        load_preset("competition")


def test_generated_workload_scales_and_remains_reproducible(project):
    path = project / "configs/runs/run_test.yaml"
    original = resolve_run(path)
    assert original.workload_sha256 == resolve_run(path).workload_sha256
    profile_path = project / "configs/workloads/workload_test.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["profile"]["jobs_per_order"] = 7
    profile_path.write_text(yaml.safe_dump(profile))
    larger = resolve_run(path)
    assert len(larger.workload.orders[0].jobs) == 7
    assert len(larger.workload.operations) == 14
    assert larger.workload_sha256 != original.workload_sha256
    assert larger.workload_sha256 == resolve_run(path).workload_sha256
    result = run_one(larger).simulation_result
    assert len(result.schedule) == 14


def test_realized_json_reuses_workload_and_complete_spt_result(project):
    path = project / "configs/runs/run_test.yaml"
    generated = resolve_run(path)
    first = run_one(generated)
    shutil.copyfile(
        first.run_dir / "realized_instance.json",
        project / "configs/workloads/workload_test.json",
    )
    scenario_path = project / "configs/scenarios/scenario_test.yaml"
    scenario = yaml.safe_load(scenario_path.read_text())
    scenario["workload"] = {
        "kind": "instance",
        "path": "../workloads/workload_test.json",
    }
    scenario_path.write_text(yaml.safe_dump(scenario))
    fixed = resolve_run(path)
    assert fixed.workload == generated.workload
    assert fixed.workload_sha256 == generated.workload_sha256
    assert fixed.provenance == generated.provenance
    assert run_one(fixed).simulation_result == first.simulation_result
    run = yaml.safe_load(path.read_text())
    run["seed"] = 99
    path.write_text(yaml.safe_dump(run))
    assert resolve_run(path).workload_sha256 == generated.workload_sha256


def test_readme_commands_and_relocated_references(project, monkeypatch, capsys):
    monkeypatch.chdir(project.parent)
    path = str(project / "configs/runs/run_test.yaml")
    assert main(["validate", "--config", path]) == 0
    assert "valid workload_sha256=" in capsys.readouterr().out
    assert main(["show-config", "--config", path]) == 0
    capsys.readouterr()
    assert not (project / "runs").exists()
    assert main(["run", "--config", path]) == 0
    assert "completed makespan=" in capsys.readouterr().out
    records = list((project / "runs").glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "completed"
    evidence = records[0].parent / record["paths"]["evaluation"]
    assert (evidence / "realized_instance.json").is_file()


def test_shared_factory_references_resolve_after_rename():
    for path in (ROOT / "configs/scenarios").glob("*.yaml"):
        scenario = yaml.safe_load(path.read_text())
        if scenario.get("factory") == "../factories/factory_test.yaml":
            # Exercise the ordinary run resolver for every affected recipe.
            for run_path in (ROOT / "configs/runs").glob("*.yaml"):
                run = yaml.safe_load(run_path.read_text())
                if run.get("scenario") == f"../scenarios/{path.name}":
                    resolved = resolve_run(run_path)
                    assert len(resolved.factory.machines) == 2
