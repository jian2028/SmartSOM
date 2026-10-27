"""Native episode transaction, recovery, scheduling and isolated CLI contracts."""

import json
import subprocess
import sys

import pytest

from smartsom.experiments import social_checkpoint as cp
from smartsom.experiments import social_long as sl
from smartsom.experiments import social_schedule as schedule

pytest.importorskip("torch")
from smartsom.config.base_diagnostic import BaseAlgorithm  # noqa: E402
from smartsom.learning.base_ppo import BasePPO  # noqa: E402


def test_checkpoint_authority_tuple_partial_tail_and_early(tmp_path):
    policy = BasePPO(BaseAlgorithm(schema="smartsom.base-algorithm/v1", width=8), 4)
    row = {"episodes": 1, "physical_ticks": 12, "action": ("a", ("b", 2))}
    state = {
        "episodes": 1,
        "physical_ticks": 12,
        "updates": 0,
        "last_episode": row,
        "early": {
            "episodes": 1,
            "physical_ticks": 12,
            "updates": 0,
            "threshold": 10,
            "overshoot": 2,
        },
    }
    cp.save(policy, tmp_path / "last.pt", state)
    # Crash immediately after last save, before any sidecars.
    cp.reconcile(tmp_path, state, 10)
    original = cp.digest(tmp_path / "early-160k.pt")
    assert original == cp.digest(tmp_path / "last.pt")
    assert cp.read(tmp_path / "early-160k.json")["overshoot"] == 2
    # Torch tuple versus decoded JSON list is a real persisted regression.
    cp.reconcile(tmp_path, state, 10)
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1
    with (tmp_path / "episodes.jsonl").open("ab") as f:
        f.write(b'{"episodes":2')
    cp.reconcile(tmp_path, state, 10)
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1
    state.update(
        episodes=2,
        physical_ticks=24,
        last_episode={"episodes": 2, "physical_ticks": 24},
    )
    cp.save(policy, tmp_path / "last.pt", state)
    cp.reconcile(tmp_path, state, 10)
    assert cp.digest(tmp_path / "early-160k.pt") == original
    assert cp.digest(tmp_path / "last.pt") != original
    (tmp_path / "early-160k.pt").unlink()
    with pytest.raises(ValueError, match="refusing later substitute"):
        cp.reconcile(tmp_path, state, 10)


def test_malformed_and_conflicting_ledger_fail_closed(tmp_path):
    policy = BasePPO(BaseAlgorithm(schema="smartsom.base-algorithm/v1", width=8), 4)
    state = {
        "episodes": 1,
        "physical_ticks": 1,
        "updates": 0,
        "early": None,
        "last_episode": {"episodes": 1, "value": 7},
    }
    cp.save(policy, tmp_path / "last.pt", state)
    (tmp_path / "episodes.jsonl").write_text("{bad}\n")
    with pytest.raises(json.JSONDecodeError):
        cp.reconcile(tmp_path, state, 10)
    (tmp_path / "episodes.jsonl").write_text('{"episodes":1,"value":8}\n')
    with pytest.raises(ValueError, match="mismatch"):
        cp.reconcile(tmp_path, state, 10)


def test_shared_admission_reserves_old_promises():
    control = {"workers": 4, "total_limit": 28}
    sample = {
        "scientific": list(range(11)),
        "reserved_idle_legacy": 13,
        "available_gib": 100,
        "cpu_percent": 35,
        "cpus": 32,
    }
    assert schedule.can_admit(sample, 0, control)
    sample["scientific"] = list(range(15))
    assert not schedule.can_admit(sample, 4, control)
    assert not schedule.can_admit(sample, 0, control)
    sample["scientific"] = list(range(11))
    sample["available_gib"] = 15.9
    assert not schedule.can_admit(sample, 0, control)
    sample.update(available_gib=100, cpu_percent=90)
    assert not schedule.can_admit(sample, 0, control)


def cli(root, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.social_long",
            "--root",
            str(root),
            *args,
        ],
        cwd=sl.ROOT,
        env={**sl.environment(), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=240,
    )


def test_real_cli_fresh_process_two_blocks(tmp_path):
    root = tmp_path / "probe"
    result = cli(root, "init", "--engineering")
    assert result.returncode == 0, result.stderr
    for target in (10, 20):
        result = cli(
            root, "train", "--arm", "all", "--seed", "4101", "--target", str(target)
        )
        assert result.returncode == 0, result.stderr
        state = cp.read(root / "train/all-4101/state.json")
        assert state["episodes"] == target
        assert state["updates"] > 0
        if target == 10:
            first = state
            early = cp.digest(root / "train/all-4101/early-160k.pt")
    assert state["updates"] > first["updates"]
    assert state["physical_ticks"] > first["physical_ticks"]
    assert cp.digest(root / "train/all-4101/early-160k.pt") == early
    rows = [
        json.loads(x)
        for x in (root / "train/all-4101/episodes.jsonl").read_text().splitlines()
    ]
    assert [r["episodes"] for r in rows] == list(range(1, 21))
    assert all(r["losses"] for r in rows)
    training, evaluation, _ = schedule.queue(root, cp.read(root / "manifest.json"))
    assert len(training) == 26
    assert all(t["split"] == "validation" for t in evaluation)
    assert {t["label"] for t in evaluation if t["method"] == "ppo"} == {
        "initial",
        "ep20",
        "early-160k",
    }
    assert cli(root, "report").returncode == 0
    assert (root / "dashboard.html").exists()


def test_worker_failure_persists_without_controller(tmp_path, monkeypatch):
    def invalid(_):
        raise ValueError("scientific input failure")

    monkeypatch.setattr(sl, "verify", invalid)
    with pytest.raises(ValueError, match="scientific input failure"):
        sl.train(tmp_path, "all", 4101, 10)
    receipt = cp.read(tmp_path / "failures/all-4101.json")
    assert "scientific input failure" in receipt["error"]
    monkeypatch.setattr(schedule, "verify", lambda _: {})
    with pytest.raises(RuntimeError, match="previous failure"):
        schedule.coordinate(tmp_path)
    # Contending worker does not own the operation and must not poison the run.
    directory = tmp_path / "train/base-4101"
    with sl.lock_file(directory / "worker.lock"):
        with pytest.raises(BlockingIOError):
            with sl.worker_guard(tmp_path, directory):
                pytest.fail("must not enter a competing worker")
    assert not (tmp_path / "failures/base-4101.json").exists()
