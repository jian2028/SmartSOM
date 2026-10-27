"""Real Base continuation and detached-process contracts."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/base-diagnostic/small.yaml"


@pytest.fixture
def calibrated(tmp_path):
    from smartsom import api

    shared = [
        "workload.orders=8",
        "workload.initial_orders=4",
        "workload.mean_interarrival=4",
        "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        "algorithm.epochs=1",
        "experiment.run.seconds=120",
    ]
    calibration = tmp_path / "calibration"
    api.run_experiment(
        CONFIG,
        task="calibrate",
        sets=shared + [f'experiment.run.output="{calibration}"'],
    )
    return shared + [f'experiment.run.calibration="{calibration / "calibration.json"}"']


def test_complete_update_recovery_is_exact_and_bounded(
    tmp_path, calibrated, monkeypatch
):
    import torch

    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime

    original = runtime.episode
    calls = 0
    interrupted = False

    def episode(*args, **kwargs):
        nonlocal calls
        kwargs["max_ticks"] = min(40, kwargs["max_ticks"])
        result = original(*args, **kwargs)
        calls += 1
        if interrupted and calls == 2:
            raise RuntimeError("injected after rollout before commit")
        return result

    monkeypatch.setattr(runtime, "episode", episode)
    for name in ("uninterrupted", "recovered"):
        calls = 0
        interrupted = name == "recovered"
        result = api.run_experiment(
            CONFIG,
            task="train",
            sets=calibrated
            + [
                f'experiment.run.output="{tmp_path / name}"',
                "experiment.train_ticks=120",
            ],
        )
        assert result["physical_ticks"] == 120 and result["episodes"] == 3
    a = torch.load(tmp_path / "uninterrupted/last.pt", weights_only=False)
    b = torch.load(tmp_path / "recovered/last.pt", weights_only=False)

    def equal(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    equal(a, b)
    assert len(json.loads((tmp_path / "recovered/attempts.json").read_text())) == 1
    from smartsom.experiments.reservation import execute_saved

    assert execute_saved(tmp_path / "recovered")["episodes"] == 3
    path = tmp_path / "recovered/inputs/algorithm.yaml"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="identity"):
        execute_saved(tmp_path / "recovered")


def test_background_launch_completes_with_frozen_inputs(tmp_path):
    root = tmp_path / "background"
    command = [
        sys.executable,
        "-m",
        "smartsom.experiments.cli",
        "run",
        str(CONFIG),
        "--background",
        "--set",
        f'experiment.run.output="{root}"',
        "--set",
        "experiment.run.methods=[rule]",
        "--set",
        "experiment.run.smoke_ticks=20",
        "--progress",
        "off",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (root / "progress.json").exists():
            value = json.loads((root / "progress.json").read_text())
            if value["status"] in ("completed", "failed"):
                break
        time.sleep(0.1)
    assert value["status"] == "completed"
    assert (root / "control/background-request.json").exists()
    assert (root / "logs/stdout.log").exists()


def test_stop_is_not_automatically_retried(tmp_path, calibrated, monkeypatch):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    original = runtime.episode

    def stopped(*args, **kwargs):
        raise StopRequested("test stop")

    monkeypatch.setattr(runtime, "episode", stopped)
    root = tmp_path / "stopped"
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task="train",
            sets=calibrated
            + [f'experiment.run.output="{root}"', "experiment.train_ticks=40"],
        )
    assert len(json.loads((root / "attempts.json").read_text())) == 1
    manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setattr(runtime, "episode", original)
    assert execute_saved(root)["physical_ticks"] == 40
    assert json.loads((root / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("task,total", [("calibrate", 8), ("evaluate", 8)])
def test_stage_cases_resume_without_repeating_commits(
    tmp_path, calibrated, monkeypatch, task, total
):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    root = tmp_path / (task + "-resume")
    original = runtime.episode
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise StopRequested("interrupt fourth case")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "episode", interrupted)
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task=task,
            sets=calibrated
            + [
                f'experiment.run.output="{root}"',
                "experiment.validation_cases=4",
                "experiment.run.methods=[rule,random]",
            ],
        )
    receipts = sorted((root / "case-commits").glob("case-*.json"))
    assert len(receipts) == 3
    before = {p.name: p.read_bytes() for p in receipts}
    manifest = (root / "manifest.json").read_bytes()
    result = execute_saved(root)
    assert result["status"] in {"completed", "frozen"}
    assert calls == total + 1
    assert (root / "manifest.json").read_bytes() == manifest
    assert all(
        (root / "case-commits" / name).read_bytes() == data
        for name, data in before.items()
    )
    assert len(list((root / "case-commits").glob("case-*.json"))) == total


def test_evaluation_expired_deadline_cannot_reset_on_resume(
    tmp_path, calibrated, monkeypatch
):
    from smartsom import api
    from smartsom.experiments import reservation_runtime as runtime
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.reservation import execute_saved

    root = tmp_path / "expired"
    original = runtime.episode

    def stopped(*args, **kwargs):
        raise StopRequested("before first case")

    monkeypatch.setattr(runtime, "episode", stopped)
    with pytest.raises(StopRequested):
        api.run_experiment(
            CONFIG,
            task="evaluate",
            sets=calibrated
            + [
                f'experiment.run.output="{root}"',
                "experiment.validation_cases=4",
                "experiment.run.methods=[rule,random]",
            ],
        )
    manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setattr(runtime, "episode", original)
    monkeypatch.setattr(runtime.time, "time", lambda: manifest["started_unix"] + 121)
    with pytest.raises(TimeoutError):
        execute_saved(root)
    assert not list((root / "case-commits").glob("case-*.json"))
    assert json.loads((root / "manifest.json").read_text()) == manifest


def test_episode_budget_validation_selection_and_periodic_checkpoint(
    tmp_path, calibrated
):
    from smartsom import api

    root = tmp_path / "episodes"
    result = api.run_experiment(
        CONFIG,
        task="train",
        sets=calibrated
        + [
            f'experiment.run.output="{root}"',
            "experiment.train_episodes=2",
            "experiment.condition_sampling=balanced_cycle",
            "experiment.validation_cases=4",
            "experiment.validation.milestones=[1,2]",
            "experiment.validation.sample_action_seeds=1",
            "experiment.validation.selection=sample_complete_cost",
            "experiment.checkpoint_every_episodes=2",
        ],
    )
    assert result["status"] == "completed" and result["episodes"] == 2
    assert result["episode_budget"] == 2 and result["tick_budget"] is None
    rows = [
        json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["condition"] for row in rows] == ["normal", "low"]
    assert (root / "checkpoints/episode-00000002.pt").is_file()
    assert not (root / "checkpoints/episode-00000001.pt").exists()
    selection = json.loads((root / "selection.json").read_text())
    assert selection["status"] == "selected"
    assert selection["selected_episode"] in (1, 2)
    receipts = {p: p.read_bytes() for p in (root / "validation").rglob("cases.jsonl")}
    assert len(receipts) == 4
    assert api.resume(root)["episodes"] == 2
    assert all(p.read_bytes() == content for p, content in receipts.items())
