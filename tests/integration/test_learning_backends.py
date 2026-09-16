"""Required learning CI runs the real fixed-budget train/save/load/evaluate gate."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smartsom.config.experiment import load_config, prepare
from smartsom.experiments import train_one
from smartsom.experiments.training_audit import audit_training

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    missing = [
        name
        for name in ("torch", "ray", "gymnasium", "sb3_contrib")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        if os.environ.get("SMARTSOM_REQUIRE_LEARNING") == "1":
            pytest.fail(f"learning acceptance requires locked extras: {missing}")
        pytest.skip("optional learning extras are not installed")
    directory = tmp_path_factory.mktemp("fixed-learning")
    results = {}
    for name in ("sb3", "rllib"):
        config = load_config(ROOT / f"configs/runs/learning_{name}.yaml")
        config.output.root = str(directory / name)
        config.logging.tensorboard = False
        config.logging.progress = "off"
        config.logging.verbose = False
        config.validation.enabled = False
        resolved = prepare(config)
        result = train_one(resolved)
        results[name] = result
    return directory, results


def test_actual_parameter_updates_restoration_and_episode_replay(trained):
    _, results = trained
    for name, steps in (("sb3", 1024), ("rllib", 4096)):
        result = results[name]
        report = audit_training(result.run_dir)
        assert report["status"] == "passed" and report["environment_steps"] == steps
        manifest = json.loads((result.checkpoint_dir / "checkpoint.json").read_text())
        assert manifest["initial_weights_sha256"] != manifest["final_weights_sha256"]
        assert manifest["learner_updates"] > 0


def test_fixed_acceptance_rejects_swapped_backend_attempts(trained):
    _, results = trained
    command = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, 'scripts'); "
        "from validate_learning import validate_fixed_training; "
        "from pathlib import Path; "
        "validate_fixed_training('rllib', Path(sys.argv[1]))",
        str(results["sb3"].run_dir),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "differs from the fixed acceptance recipe" in result.stderr


def test_fifteen_fixed_paired_runs_are_audited_without_false_completion(trained):
    directory, results = trained
    output = directory / "evaluation"
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_learning.py"),
        "--development",
        "--rllib-training-dir",
        str(results["rllib"].run_dir),
        "--sb3-training-dir",
        str(results["sb3"].run_dir),
        "--output-dir",
        str(output),
        "--workers",
        "2",
    ]
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=240
    )
    report = json.loads((output / "report.json").read_text())
    assert len(report["evaluation"]) == 15
    for replication in range(5):
        rows = [r for r in report["evaluation"] if r["replication"] == replication]
        assert {r["provider"] for r in rows} == {
            "builtin.spt",
            "rllib.ppo",
            "sb3.maskable_ppo",
        }
        assert len({r["world_sha256"] for r in rows}) == 1
    for row in report["evaluation"]:
        assert row["execution_replay"]["status"] in {"passed", "partial_verified"}
        if row["provider"] == "builtin.spt":
            assert row["audit_status"] == "passed"
        if row["audit_status"] != "passed":
            assert row["makespan"] is None
            assert (
                row["run_status"] != "completed" or "qualified demand" in row["error"]
            )
    all_complete = all(row["audit_status"] == "passed" for row in report["evaluation"])
    assert report["status"] == ("passed" if all_complete else "failed")
    assert result.returncode == (0 if all_complete else 1)
    assert (
        report["evidence_kind"] == "development"
        and report["accepted_source_sha"] is None
    )
