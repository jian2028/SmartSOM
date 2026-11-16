"""Tune lifecycle publication preserves scientific budgets and bounded writes."""

import json
from types import SimpleNamespace

from smartsom.experiments.tune_progress import ScientificProgress
from smartsom.telemetry.runtime import CURRENT
from tests.unit.test_runtime_display import display


def test_completion_before_status_transition_and_retry(tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "episodes": 2,
                "episode_budget": 8,
                "physical_ticks": 120,
                "status": "wall_budget_reached",
            }
        )
    )
    trial = SimpleNamespace(
        trial_id="one",
        config={"root": str(tmp_path), "task": "train"},
        status="RUNNING",
    )
    view, _ = display(tmp_path)
    callback = ScientificProgress()
    token = CURRENT.set(view)
    try:
        callback.on_trial_complete(1, [trial], trial)
        task = view.snapshot()["tasks"][0]
        assert (task["completed"], task["total"], task["unit"]) == (2, 8, "episodes")
        assert (
            json.loads((tmp_path / "tune-status.json").read_text())["status"]
            == "completed"
        )
        callback.on_trial_start(2, [trial], trial)
        callback.on_experiment_end([trial])
        assert (
            json.loads((tmp_path / "tune-status.json").read_text())["status"]
            == "running"
        )
    finally:
        CURRENT.reset(token)


def test_unchanged_terminal_trials_do_not_republish(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("{}")
    callback = ScientificProgress()
    trial = SimpleNamespace(
        trial_id="one",
        config={"root": str(tmp_path), "task": "train"},
        status="TERMINATED",
    )
    writes = []
    monkeypatch.setattr(
        "smartsom.experiments.tune_progress.write_json",
        lambda *args: writes.append(args),
    )
    callback.on_step_end(0, [trial])
    callback.on_experiment_end([trial])
    assert len(writes) == 1
