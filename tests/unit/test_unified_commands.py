"""Explicit tasks validate the selected operation, not training-field presence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom import api
from smartsom.experiments.cli import _parser, main
from smartsom.experiments.commands import resolve

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "configs/test/runs/evaluate_all_rules.yaml"
TRAIN = ROOT / "configs/test/runs/train_machine_ppo.yaml"


def test_check_rule_evaluation_allocates_nothing(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("check cannot execute")

    monkeypatch.setattr(api, "evaluate", forbidden)
    assert (
        main(
            [
                "check",
                "--task",
                "evaluate",
                "--config",
                str(RULES),
                "--output-root",
                str(tmp_path / "absent"),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["training"] is None
    assert not any(group["trainable"] for group in plan["groups"].values())
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "missing"],
        ["--preview", "--replications", "1"],
        ["--preview", "--replay"],
        ["--preview", "--scenario", "missing"],
        ["--mode", "balanced"],
        ["--steps", "10"],
        ["--set", "seed=12"],
    ],
)
def test_explicit_entry_rejects_ambiguous_or_unsupported_options(extra):
    assert main(["check", "--task", "evaluate", "--config", str(RULES), *extra]) == 2


def test_training_recipe_cannot_be_evaluated_as_fresh_weights():
    assert main(["check", "--task", "evaluate", "--config", str(TRAIN)]) == 2


@pytest.mark.parametrize(
    "task,method", [("train", "train"), ("train-evaluate", "train_evaluate")]
)
def test_explicit_training_dispatch(task, method, monkeypatch, capsys):
    from smartsom.experiments import commands

    config = api.load_config(TRAIN)
    monkeypatch.setattr(
        commands, "resolve", lambda args: ({"input_type": "experiment-v3"}, config)
    )
    called = []
    monkeypatch.setattr(
        api, method, lambda actual: called.append(actual) or {"status": "completed"}
    )
    assert main(["run", "--task", task, "--config", str(TRAIN)]) == 0
    assert called == [config]
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_check_checkpoint_conditions_precede_execution(monkeypatch):
    config = api.load_config(TRAIN)
    config.checkpointing.save_last = False
    monkeypatch.setattr(
        "smartsom.experiments.commands.load_config", lambda path: config
    )
    args = _parser().parse_args(
        ["check", "--task", "train-evaluate", "--config", str(TRAIN)]
    )
    with pytest.raises(ValueError, match="save_last"):
        resolve(args)


def test_preview_is_one_case_recorded_and_marked(tmp_path, capsys, monkeypatch):
    config = api.load_config(RULES)
    config.scenario_overrides["tick_limit"] = 8
    monkeypatch.setattr(
        "smartsom.experiments.commands.load_config", lambda path: config
    )
    assert (
        main(
            [
                "run",
                "--task",
                "evaluate",
                "--preview",
                "--config",
                str(RULES),
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    root = Path(result["run_dir"])
    assert len(result["results"]) == 1
    assert json.loads((root / "run.json").read_text())["purpose"] == "preview"
    assert (root / "evidence/case-0000/trace.jsonl").is_file()


def test_resume_routes_by_saved_type(tmp_path, monkeypatch):
    from smartsom.experiments.commands import resume

    calls = []
    args = SimpleNamespace(source=tmp_path, retry_failed=True)
    monkeypatch.setattr(api, "resume_tune_batch", lambda *a, **k: calls.append("tune"))
    monkeypatch.setattr(api, "run_study", lambda *a, **k: calls.append("study"))
    (tmp_path / "batch.json").write_text("{}")
    (tmp_path / "plan.json").write_text('{"schema": "smartsom.tune-batch/v1"}')
    resume(args)
    (tmp_path / "batch.json").unlink()
    (tmp_path / "study.json").write_text("{}")
    (tmp_path / "plan.json").write_text(
        '{"schema": "smartsom.composable-study-plan/v1"}'
    )
    resume(args)
    assert calls == ["tune", "study"]


def test_legacy_run_does_not_silently_accept_task_flags():
    assert main(["run", "--config", str(RULES), "--preview"]) == 2


def test_tune_schema_dispatch_and_task_rejection(tmp_path, monkeypatch, capsys):
    batch = tmp_path / "batch.yaml"
    batch.write_text("schema: smartsom.tune-batch/v1\n")
    calls = []

    def tune(**kwargs):
        calls.append(kwargs)
        return {"status": "feasible"}

    monkeypatch.setattr(api, "tune_batch", tune)
    assert main(["check", "--task", "train-evaluate", "--config", str(batch)]) == 0
    assert calls[0]["action"] == "check"
    assert json.loads(capsys.readouterr().out)["executor"] == "ray-tune"
    assert main(["check", "--task", "train", "--config", str(batch)]) == 2
    assert len(calls) == 1


def test_prepared_study_check_never_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        "smartsom.experiments.composable_study.show_study",
        lambda root: {"status": "prepared"},
    )
    monkeypatch.setattr(
        api, "run_study", lambda *a, **k: pytest.fail("check executed Study")
    )
    assert main(["check", "--task", "train-evaluate", "--study", "prepared"]) == 0
    assert json.loads(capsys.readouterr().out)["executor"] == "native-study"
    assert main(["check", "--task", "evaluate", "--study", "prepared"]) == 2


def test_render_and_record_options_reach_v3_execution(tmp_path, monkeypatch, capsys):
    observed = []

    def live(factory, controls, thread):
        controls.delay = 0
        thread.start()
        thread.join()
        observed.append(controls.latest)

    monkeypatch.setitem(
        sys.modules, "smartsom.studio.playback", SimpleNamespace(live_window=live)
    )
    config = api.load_config(RULES)
    config.scenario_overrides["tick_limit"] = 8
    monkeypatch.setattr(
        "smartsom.experiments.commands.load_config", lambda path: config
    )
    assert (
        main(
            [
                "run",
                "--task",
                "evaluate",
                "--preview",
                "--config",
                str(RULES),
                "--render-mode",
                "human",
                "--no-record",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert observed[0]["tick"] == 8
    assert not (Path(result["run_dir"]) / "evidence/case-0000/trace.jsonl").exists()
