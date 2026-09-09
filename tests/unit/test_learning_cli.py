"""Learning orchestration CLI preserves explicit inputs and process outcomes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.experiments.cli import main
from smartsom.experiments.learning_study import LearningStudyResult


def outcome(status="completed", failed=0):
    return LearningStudyResult(Path("runs/study"), status, 2, failed, 0, ())


def test_batch_cli_expands_recipes_and_seeds_once(tmp_path, monkeypatch, capsys):
    files = [tmp_path / f"{name}.yaml" for name in ("sb3_micro", "marl_micro")]
    for path in files:
        path.write_text(f"preset: {path.stem}\n")
    captured = {}

    def batch(configs, **kwargs):
        captured.update(configs=configs, **kwargs)
        return outcome()

    monkeypatch.setattr(api, "batch_train", batch)
    assert (
        main(
            [
                "batch-train",
                "--recipe",
                str(files[0]),
                "--recipe",
                str(files[1]),
                "--seeds",
                "101",
                "102",
                "--max-concurrent",
                "2",
                "--steps",
                "128",
                "--steps-per-update",
                "64",
            ]
        )
        == 0
    )
    assert [config.seed for config in captured["configs"]] == [101, 102, 101, 102]
    assert all(config.training.total_steps == 128 for config in captured["configs"])
    assert captured["max_concurrent"] == 2
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seed", "101", "--seeds", "102"],
        ["--seeds", "101", "101"],
        ["--resume", "old", "--steps", "128"],
        ["--retry-failed"],
    ],
)
def test_batch_cli_rejects_ambiguous_or_implicit_retries(arguments, monkeypatch):
    monkeypatch.setattr(api, "batch_train", lambda *a, **k: pytest.fail("executed"))
    assert main(["batch-train", "--preset", "sb3_micro", *arguments]) == 2


def test_search_cli_preserves_frozen_resume(monkeypatch, capsys):
    captured = {}

    def search(**kwargs):
        captured.update(kwargs)
        return outcome("completed_with_failures", 1)

    monkeypatch.setattr(api, "search", search)
    assert main(["search", "--resume", "saved", "--retry-failed"]) == 1
    assert captured == {"resume": Path("saved"), "retry_failed": True}
    assert json.loads(capsys.readouterr().out)["failed"] == 1


def test_search_cli_strict_nested_overrides(monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(
        api, "search", lambda config: captured.append(config) or outcome()
    )
    assert (
        main(
            [
                "search",
                "--preset",
                "sb3_micro",
                "--set",
                "search.space={algorithm.learning_rate: {type: categorical, choices: [0.0001, 0.0003]}}",
                "--set",
                "search.objective=makespan",
                "--set",
                "search.failure_policy=all_complete",
            ]
        )
        == 0
    )
    assert captured[0].search.space["algorithm.learning_rate"].choices == (
        0.0001,
        0.0003,
    )
    capsys.readouterr()


@pytest.mark.parametrize(
    "status,code", [("completed", 0), ("failed", 1), ("interrupted", 130)]
)
def test_study_exit_code_reflects_actual_result(status, code, monkeypatch):
    monkeypatch.setattr(
        api, "search", lambda *a, **k: replace(outcome(), status=status)
    )
    assert main(["search", "--preset", "sb3_micro"]) == code
