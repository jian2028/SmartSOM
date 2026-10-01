"""End-to-end disposable batch with real observation, probes and native learning."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from test_tuning_probe_native import (
    frozen_engineering_input,
    require_optional_cpu_runtime,
)

from smartsom.experiments import tuning_batch as batch
from smartsom.experiments.tuning_session import verify_identity

pytestmark = pytest.mark.learning


def test_cli_check_import_is_json_and_does_not_allocate_or_mutate(tmp_path):
    require_optional_cpu_runtime()
    from smartsom.config.codec import digest
    from smartsom.experiments.composable import archive_inputs
    from smartsom.experiments.evidence import write_json

    prepared, _ = frozen_engineering_input(tmp_path, "ppo")
    root = tmp_path / "prepared-study"
    (root / "snapshots/a/config").mkdir(parents=True)
    frozen = archive_inputs(root / "snapshots/a", prepared)
    snapshot = root / "snapshots/a/config/prepared.json"
    plan = {
        "schema": "smartsom.composable-study-plan/v1",
        "source": {"old": True},
        "recipe": {"controls": ["initial", "rule", "random"]},
        "entries": [
            {
                "id": "a",
                "snapshot": "snapshots/a/config/prepared.json",
                "snapshot_sha256": digest(json.loads(snapshot.read_text())),
                "scientific_sha256": frozen.scientific_sha256,
            }
        ],
    }
    write_json(root / "plan.json", plan)
    write_json(root / "study.json", {"plan_sha256": digest(plan)})
    before = {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "tune",
            "check",
            "--study",
            str(root),
            "--log-format",
            "json",
            "--progress",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "feasible"
    assert not (tmp_path / "tune-runs").exists()
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    ("algorithm", "execution"), [("ppo", "fixed"), ("dqn", "adaptive")]
)
def test_real_batch_calibrates_executes_and_skips_completed_resume(
    tmp_path, algorithm, execution
):
    require_optional_cpu_runtime()
    import ray

    prepared, _ = frozen_engineering_input(tmp_path, algorithm)
    inputs = batch.BatchInputs(
        (
            {
                "experiment_id": algorithm,
                "prepared": asdict(prepared),
                "control_spec": {},
            },
        ),
        mode="performance",
        execution=execution,
        active_limit=45,
        output_root=str(tmp_path / "engineering-only"),
    )
    checked = batch.preflight(inputs)
    assert checked["status"] == "feasible"
    assert not Path(inputs.output_root).exists() and not ray.is_initialized()
    root, plan, state = batch.allocate_batch(inputs)
    result = batch.execute_batch(root, plan, state)
    assert result["status"] == "completed", json.loads(
        (root / "batch.json").read_text()
    )
    assert result["completed"] == 1 and result["failed"] == 0
    assert not ray.is_initialized()
    calibration = json.loads((root / "calibration.json").read_text())
    assert 0 < calibration["active_seconds"] <= 45
    assert calibration["recommendations"] and not calibration["missing_groups"]
    assert any(
        m["valid"] and m["stages"]["updates"] >= 1 for m in calibration["measurements"]
    )
    saved = json.loads((root / "batch.json").read_text())
    row = saved["entries"][algorithm]
    assert row["physical_ticks"] == 8 and row["updates"] == 2
    record = json.loads((Path(row["checkpoint"]) / "record.json").read_text())
    batch.load_run(root)
    frozen = row.get("selected_prepared", plan["entries"][0]["prepared"])
    from smartsom.config.experiment_v3 import PreparedComposition

    marker = verify_identity(PreparedComposition(**frozen), record, row["checkpoint"])
    assert marker["phase"] == "experiment_complete"
    before = (root / "batch.json").read_bytes()
    assert batch.resume_batch(root)["completed"] == 1
    assert (root / "batch.json").read_bytes() == before
    assert not ray.is_initialized()
