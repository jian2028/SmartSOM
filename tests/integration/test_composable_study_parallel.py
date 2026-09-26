"""Actual bounded process execution with real PPO/DQN and frozen inputs."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json, digest
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments import composable_study as study
from smartsom.experiments.composable import archive_inputs, implementation_identity
from smartsom.experiments.evidence import source_identity

ROOT = Path(__file__).resolve().parents[2]


def _freeze_study(tmp_path, concurrency, *, count=8, total_ticks=32):
    for folder in ("experiments", "controls", "snapshots"):
        (tmp_path / folder).mkdir()
    entries = []
    for index in range(count):
        algorithm = "ppo" if index % 2 else "dqn"
        config = api.load_config(ROOT / f"configs/test/runs/train_all_{algorithm}.yaml")
        config.composition = str(
            ROOT / f"configs/test/compositions/small_train_{algorithm}.yaml"
        )
        config.scenario = str(ROOT / "configs/test/scenarios/small_matrix_zero.yaml")
        config.scenario_overrides["tick_limit"] = 12
        config.seed = 101 + index
        config.training.groups = ("machine", "buffer", "dispatcher")
        config.training.total_ticks = total_ticks
        config.training.ticks_per_update = 16
        config.training.record_initial = True
        config.validation.enabled = True
        config.validation.every_updates = 1
        config.validation.replications = 2
        config.evaluation.replications = 2
        config.evaluation.record = False
        config.evaluation.full_replay = True
        config.output.root = str(tmp_path / "experiments")
        config.output.name = f"child-{index}"
        prepared = prepare_v3(config)
        parameters = json.loads(prepared.parameters_json)
        parameters.update(batch_size=2)
        if algorithm == "ppo":
            parameters.update(n_epochs=1)
        else:
            parameters.update(warmup_ticks=0, target_update_ticks=8)
        prepared = replace(prepared, parameters_json=canonical_json(parameters))
        name = f"h0_low_{algorithm}_zero_{index}"
        snapshot_root = tmp_path / "snapshots" / name
        (snapshot_root / "config").mkdir(parents=True)
        archive_inputs(snapshot_root, prepared)
        snapshot = snapshot_root / "config/prepared.json"
        entries.append(
            {
                "id": name,
                "H_case": "h0",
                "H": 0,
                "V_case": "low",
                "V": 0.2,
                "algorithm": algorithm,
                "transport": "zero",
                "snapshot": str(snapshot.relative_to(tmp_path)),
                "snapshot_sha256": digest(json.loads(snapshot.read_text())),
            }
        )
    plan = {
        "recipe": {
            "max_concurrent": concurrency,
            "total_ticks": total_ticks,
            "ticks_per_update": 16,
            "validation_every_updates": 1,
            "validation_cases": 2,
            "evaluation_cases": 2,
            "controls": ["initial", "rule", "random"],
        },
        "implementation_sha256": implementation_identity(),
        "source": source_identity(),
        "H": {},
        "datasets": {"test_000": {"V": {"low": 0.3}}},
        "entries": entries,
    }
    study._json(tmp_path / "plan.json", plan)
    study._json(
        tmp_path / "study.json",
        {
            "plan_sha256": digest(plan),
            "status": "prepared",
            "entries": {},
            "controls": {},
        },
    )
    return entries


@pytest.mark.learning
@pytest.mark.parametrize("concurrency", [6, 8])
def test_real_experiments_overlap_and_restart_skips_completed(tmp_path, concurrency):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    entries = _freeze_study(tmp_path, concurrency)
    result = study.run_study(tmp_path, display_options={"verbose": False})
    assert result["status"] == "completed"
    state = json.loads((tmp_path / "study.json").read_text())
    assert len({e["pid"] for e in state["entries"].values()}) == 8
    assert len(state["controls"]) == 2
    intervals = []
    directories = set()
    for row in entries:
        entry = state["entries"][row["id"]]
        directory = Path(entry["run_dir"])
        directories.add(directory)
        record = json.loads((directory / "run.json").read_text())
        assert record["status"] == "completed" and record["physical_ticks"] == 32
        assert sum(record["optimizations"].values()) > 0
        progress = json.loads(
            (tmp_path / "workers" / row["id"] / "logs/progress.json").read_text()
        )
        training = next(r for r in progress["tasks"] if r["id"] == "training")
        assert training["completed"] == 32
        assert training["values"]["validation_finished"] == 2
        assert training["values"]["algorithm"] == row["algorithm"].upper()
        intervals.append((entry["started_at"], entry["ended_at"]))
        assert (directory / "checkpoints/recovery.json").exists()
    assert len(directories) == 8
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak == concurrency
    # A second invocation must not allocate or update a completed child.
    before = sorted(p.name for p in (tmp_path / "experiments").iterdir())
    assert (
        study.run_study(tmp_path, display_options={"verbose": False})["status"]
        == "completed"
    )
    assert sorted(p.name for p in (tmp_path / "experiments").iterdir()) == before
    parent = json.loads((tmp_path / "logs/progress.json").read_text())
    assert parent["total_tasks"] == 8
    assert all(r["completed"] == 32 for r in parent["tasks"])


@pytest.mark.learning
def test_interrupt_stops_workers_and_resume_keeps_saved_training(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    _freeze_study(tmp_path, 2, count=2, total_ticks=256)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from smartsom.experiments.composable_study import run_study; import sys; run_study(sys.argv[1], display_options={'verbose': False})",
            str(tmp_path),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    saved = None
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and process.poll() is None:
            checkpoints = list(
                (tmp_path / "experiments").glob(
                    "*/checkpoints/update-000001/continuation.pkl"
                )
            )
            if checkpoints:
                saved = checkpoints[0].parents[2]
                break
            time.sleep(0.05)
        assert saved is not None, "worker never saved its first real update"
        process.send_signal(signal.SIGINT)
        process.wait(timeout=45)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=45)
    state = json.loads((tmp_path / "study.json").read_text())
    assert state["status"] == "interrupted"
    for entry in state["entries"].values():
        with pytest.raises(ProcessLookupError):
            os.kill(entry["pid"], 0)
    assert (saved / "checkpoints/recovery.json").exists()
    assert (
        study.run_study(tmp_path, display_options={"verbose": False})["status"]
        == "completed"
    )
    resumed = json.loads((tmp_path / "study.json").read_text())
    assert any(Path(e["run_dir"]) == saved for e in resumed["entries"].values())
    for entry in resumed["entries"].values():
        record = json.loads((Path(entry["run_dir"]) / "run.json").read_text())
        assert record["physical_ticks"] == 256 and record["updates"] == 16
