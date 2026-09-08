"""Required learning CI runs the real fixed-budget train/save/load/evaluate gate."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config import resolve_training_run
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
        resolved = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")
        resolved = replace(
            resolved,
            run=resolved.run.model_copy(update={"output_root": str(directory / name)}),
        )
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


def test_all_fifteen_fixed_paired_evaluations_and_both_replays(trained):
    directory, results = trained
    output = directory / "evaluation"
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_learning.py"),
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
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed" and report["completed"] == 15
    assert len(report["evaluation"]) == 15 and report["failed"] == 0
