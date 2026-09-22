"""Display switches do not alter actual learner state, commands or evidence."""

import json
import subprocess
import sys

import pytest
from test_training_lifecycle import assert_same_state
from test_training_sampling import checkpoint_rows, optimizer_and_rng, small_training

from smartsom.api import train_prepared


@pytest.mark.learning
@pytest.mark.parametrize("backend", ["sb3", "rllib", "marl"])
def test_display_does_not_change_real_training(backend, tmp_path):
    prepared = small_training(backend, tmp_path)
    hidden = train_prepared(prepared, display_options={"verbose": False})
    shown = train_prepared(
        prepared,
        display_options={"verbose": True, "progress": "off", "every_seconds": 0.001},
    )
    left = json.loads((hidden.last_checkpoint / "checkpoint.json").read_text())
    right = json.loads((shown.last_checkpoint / "checkpoint.json").read_text())
    assert left["final_weights_sha256"] == right["final_weights_sha256"]
    assert checkpoint_rows(hidden) == checkpoint_rows(shown)
    # Helpers use the legacy result's checkpoint_dir alias; adapt only the test view.
    from types import SimpleNamespace

    def adapted(result):
        return SimpleNamespace(
            checkpoint_dir=result.last_checkpoint,
            last_checkpoint=result.last_checkpoint,
        )

    assert_same_state(
        optimizer_and_rng(adapted(hidden), backend),
        optimizer_and_rng(adapted(shown), backend),
    )
    assert (hidden.environment_steps, hidden.learner_updates, hidden.ppo_updates) == (
        shown.environment_steps,
        shown.learner_updates,
        shown.ppo_updates,
    )
    assert (hidden.training_dir / "episodes.jsonl").read_bytes() == (
        shown.training_dir / "episodes.jsonl"
    ).read_bytes()

    def numeric_rows(result):
        return [
            json.loads(line)
            for line in (result.training_dir / "learner_metrics.jsonl")
            .read_text()
            .splitlines()
        ]

    for left_row, right_row in zip(
        numeric_rows(hidden), numeric_rows(shown), strict=True
    ):
        assert left_row.keys() == right_row.keys()
        assert left_row["metrics"].keys() == right_row["metrics"].keys()
        # Framework wall-clock timers remain recorded, but differ across runs.
        for row in (left_row, right_row):
            row["metrics"] = {
                k: v
                for k, v in row["metrics"].items()
                if "/timers/" not in k and not k.endswith("_timer")
            }
        assert left_row == right_row
    assert (
        json.loads((shown.run_dir / "logs/progress.json").read_text())["status"]
        == "completed"
    )


@pytest.mark.learning
def test_fresh_optuna_search_only_emits_parent_json(tmp_path):
    pytest.importorskip("optuna")
    command = [
        sys.executable,
        "-m",
        "smartsom.experiments.cli",
        "search",
        "--preset",
        "sb3_micro",
        "--output-root",
        str(tmp_path),
        "--log-format",
        "json",
    ]
    for value in (
        "training.total_steps=32",
        "training.steps_per_update=16",
        "training.max_decisions=32",
        "algorithm.batch_size=8",
        "algorithm.n_epochs=1",
        "algorithm.hidden_sizes=[16,16]",
        "logging.tensorboard=false",
        "validation.every_updates=2",
        "validation.replications=1",
        "search.method=optuna",
        "search.trials=1",
        "search.objective=makespan",
        "search.failure_policy=all_complete",
        'search.space={"algorithm.learning_rate":{"type":"float","low":0.0001,"high":0.0004}}',
    ):
        command.extend(("--set", value))
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode in (0, 1), result.stderr
    assert json.loads(result.stdout)["pending"] == 0
    rows = [json.loads(line) for line in result.stderr.splitlines()]
    assert rows and all(row["schema"] == "smartsom.runtime-progress/v1" for row in rows)
    assert len(rows[-1]["tasks"]) == 1
    assert "\x1b" not in result.stderr and "\r" not in result.stderr
