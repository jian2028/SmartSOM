"""Small real PPO inference, per-mode validation, and portable native recovery."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_tuning_native_continuation import run_complete, tiny, wrapper

from smartsom.config.codec import canonical_json
from smartsom.config.experiment import EvaluationOptions, ValidationOptions
from smartsom.experiments import composable
from smartsom.experiments.evaluation_modes import for_mode
from smartsom.learning.production_inference import ModelPolicy

pytestmark = pytest.mark.learning


def dual(tmp_path):
    prepared = tiny(
        "ppo", tmp_path, empty_validation=True, initial_control_replications=1
    )
    data = json.loads(prepared.config_json)
    data["validation"].update(
        action_modes=["argmax", "sampling"], updates=[1, 2], every_updates=None
    )
    data["evaluation"].update(
        action_modes=["argmax", "sampling"], checkpoint="best", record=True
    )
    return replace(prepared, config_json=canonical_json(data))


@pytest.mark.parametrize("model", [ValidationOptions, EvaluationOptions])
def test_explicit_modes_are_nonempty_unique_and_do_not_replace_bool(model):
    for modes in ((), ("argmax", "argmax"), ("unknown",)):
        with pytest.raises(ValueError):
            model(action_modes=modes)
    assert model(action_modes=("argmax", "sampling")).deterministic is True


def test_real_categorical_sampling_changes_actions_and_replays_seed(monkeypatch):
    from smartsom.learning import production_inference as inference

    class Encoder:
        def set_training(self, _):
            pass

        def encode(self, _):
            return {"x": torch.zeros(1)}

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, _):
            return torch.tensor([[0.0, 0.2, -torch.inf]]), torch.zeros(1)

    request = SimpleNamespace(
        role="machine",
        owner="m1",
        observation={},
        tick=0,
        stage="machine",
        candidates=[
            SimpleNamespace(legal=i < 2, action=i, identity=str(i)) for i in range(3)
        ],
    )
    monkeypatch.setattr(inference, "tensor_inputs", lambda *args: None)

    def actions(seed, deterministic):
        policy = ModelPolicy(Encoder(), Network(), {"algorithm": "ppo"}, seed)
        policy.deterministic = deterministic
        return [policy.choose(request).action for _ in range(64)]

    sampled = actions(131, False)
    assert set(sampled) == {0, 1}
    assert actions(131, True) == [1] * 64
    assert sampled == actions(131, False)
    assert sampled != actions(132, False)


@pytest.mark.parametrize("retained", [False, True])
def test_native_dual_validation_best_and_full_cross_evaluation_resume(
    tmp_path, monkeypatch, retained
):
    prepared = dual(tmp_path / "runs")
    if retained:
        data = json.loads(prepared.config_json)
        data["checkpointing"]["retention"] = {"mode": "latest_full_and_best"}
        prepared = replace(prepared, config_json=canonical_json(data))
    session = wrapper(prepared)
    root = session.root
    try:
        # Real PPO sampling, updates and world evaluations; controlled selector
        # exposes distinct winning candidates instead of relying on tiny scores.
        from smartsom.experiments import training_validation

        choices = iter([True, True, False, True])
        monkeypatch.setattr(
            training_validation,
            "select_best",
            lambda *args: (next(choices), "controlled rank"),
        )
        assert not session.step()["done"]
        saved = session.save_checkpoint(tmp_path / "portable")
        record = copy.deepcopy(session.record)
        record["tuning"]["continuation"] = saved
    finally:
        session.close()
    restored = wrapper(prepared, root=tmp_path / "restored", rec=record)
    try:
        assert restored.session.mode_bests["argmax"]["update"] == 1
        assert run_complete(restored)["done"]
        root = restored.root
        bests = json.loads((root / "checkpoints/best-modes.json").read_text())
        assert bests["argmax"]["update"] == 1
        assert bests["sampling"]["update"] == 2
        assert bests["argmax"]["checkpoint"] != bests["sampling"]["checkpoint"]
        manifest = json.loads((root / "evaluation/modes.json").read_text())
        assert manifest["action_modes"] == ["argmax", "sampling"]
        assert set(manifest["controls"]) == {"initial/argmax", "initial/sampling"}
        for selector, selection in manifest["selections"].items():
            assert set(selection["results"]) == {"argmax", "sampling"}
            for mode, path in selection["results"].items():
                rows = json.loads((root / path).read_text())
                assert len(rows) == 1 and rows[0]["action_mode"] == mode
                assert rows[0]["selection_mode"] == selector
                assert not rows[0]["engineering_failure"]
        assert len(list((root / "logs").glob("validation-*-argmax.json"))) == 2
        assert len(list((root / "logs").glob("validation-*-sampling.json"))) == 2
        calls = []
        original = composable.policies_for

        def counted(*args, **kwargs):
            calls.append(kwargs["deterministic"])
            return original(*args, **kwargs)

        monkeypatch.setattr(composable, "policies_for", counted)
        restored._finish_evaluation()
        assert calls == []  # every mode has independent verified case receipts
        assert json.loads((root / "checkpoints/best-modes.json").read_text()) == bests
    finally:
        restored.close()


def test_modes_refuse_other_modes_case_cache_and_seed_drift(tmp_path):
    prepared = dual(tmp_path / "runs")
    session = wrapper(prepared)
    try:
        checkpoint = composable.checkpoint_path(session.root, "last")
        recipe = composable.evaluation_recipe(prepared, checkpoint)
        cases = json.loads(prepared.evaluation_json)
        directory = tmp_path / "cases"
        composable.evaluate_cases(
            for_mode(recipe, "argmax"), cases, directory=directory
        )
        with pytest.raises(ValueError, match="identity changed"):
            composable.evaluate_cases(
                for_mode(recipe, "sampling"), cases, directory=directory
            )
        altered = copy.deepcopy(cases)
        altered[0]["policy_root"] = 123
        with pytest.raises(ValueError, match="identity changed"):
            composable.evaluate_cases(
                for_mode(recipe, "argmax"), altered, directory=directory
            )
    finally:
        session.close()


def test_interrupted_second_validation_mode_resumes_without_new_training(
    tmp_path, monkeypatch
):
    from smartsom.experiments.control import StopRequested

    prepared = dual(tmp_path / "runs")
    before = wrapper(prepared)
    original = composable.TrainingSession._validate_candidate

    def interrupted(self, checkpoint, mode=None):
        if mode == "sampling":
            raise StopRequested("injected between modes")
        return original(self, checkpoint, mode)

    monkeypatch.setattr(composable.TrainingSession, "_validate_candidate", interrupted)
    try:
        assert before.step()["done"]  # cooperative stop publishes pending validation
        assert before.session.ticks == 4
        assert before.session.mode_bests["argmax"]["evaluated_update"] == 1
        saved = before.save_checkpoint(tmp_path / "interrupted")
        record = copy.deepcopy(before.record)
        record["tuning"]["continuation"] = saved
    finally:
        before.close()
    monkeypatch.setattr(composable.TrainingSession, "_validate_candidate", original)
    after = wrapper(prepared, root=tmp_path / "resumed", rec=record)
    try:
        score = copy.deepcopy(after.session.mode_bests["argmax"])
        assert not after.step()["done"]
        assert after.session.ticks == 4 and after.session.updates == 1
        assert after.session.mode_bests["argmax"] == score
        assert after.session.mode_bests["sampling"]["evaluated_update"] == 1
    finally:
        after.close()


def test_standard_cli_saves_both_final_and_initial_and_resumes(tmp_path):
    import yaml
    from test_author_learning import _inputs

    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.experiments.author_driver import allocate
    from smartsom.experiments.cli import main

    source = _inputs(tmp_path / "inputs", "rllib")
    data = yaml.safe_load(source.read_text())
    data["validation"]["action_modes"] = ["argmax", "sampling"]
    data["evaluation"].update(action_modes=["argmax", "sampling"], initial_control=True)
    source.write_text(yaml.safe_dump(data, sort_keys=False))
    plan = compile_experiment(source, require_dependencies=True)
    assert plan.entries[0].prepared.config.evaluation.action_modes == (
        "argmax",
        "sampling",
    )
    root = allocate(plan)
    assert main(["resume", str(root), "--no-background"]) == 0
    stage = json.loads((root / "entries/entry-0001/stages.json").read_text())
    for name in ("evaluation", "initial_control"):
        directory = Path(stage["stages"][name]["run_dir"])
        record = json.loads((directory / "run.json").read_text())
        assert {row["action_mode"] for row in record["results"]} == {
            "argmax",
            "sampling",
        }
        for mode in ("argmax", "sampling"):
            assert (directory / "evidence" / mode / ".case-resume/plan.json").is_file()
    assert main(["resume", str(root), "--no-background"]) == 0


def test_loaded_policy_flags_and_sampling_case_reproduce_exactly(tmp_path, monkeypatch):
    prepared = dual(tmp_path / "runs")
    session = wrapper(prepared)
    try:
        recipe = composable.evaluation_recipe(
            prepared, composable.checkpoint_path(session.root, "last")
        )
        cases = json.loads(prepared.evaluation_json)
        original = composable.policies_for
        observed = []

        def spy(*args, **kwargs):
            policies, learners = original(*args, **kwargs)
            flags = {
                p.deterministic
                for p in policies.values()
                if hasattr(p, "deterministic")
            }
            assert flags == {kwargs["deterministic"]}
            observed.extend(flags)
            return policies, learners

        monkeypatch.setattr(composable, "policies_for", spy)
        records = []
        for name, mode in (
            ("greedy", "argmax"),
            ("sample-a", "sampling"),
            ("sample-b", "sampling"),
        ):
            rows = composable.evaluate_cases(
                for_mode(recipe, mode), cases, directory=tmp_path / name
            )
            records.append(
                [{k: v for k, v in row.items() if k != "run_dir"} for row in rows]
            )
        assert observed == [True, False, False]
        assert records[1] == records[2]
        assert (
            records[0][0]["decision_diagnostics"]
            != records[1][0]["decision_diagnostics"]
        )
    finally:
        session.close()


def test_final_mode_interruption_restores_only_unfinished_modes(tmp_path, monkeypatch):
    from smartsom.experiments.control import StopRequested

    prepared = dual(tmp_path / "runs")
    before = wrapper(prepared)
    before.record["tuning"]["evaluation_case_root"] = str(tmp_path / "durable-cases")
    real = composable.evaluate_cases
    calls = []

    def interrupt_after_commit(*args, **kwargs):
        rows = real(*args, **kwargs)
        if kwargs.get("label") == "final evaluation":
            raise StopRequested("after first final mode committed")
        return rows

    try:
        assert not before.step()["done"]
        with monkeypatch.context() as patch:
            patch.setattr(composable, "evaluate_cases", interrupt_after_commit)
            with pytest.raises(StopRequested):
                before.step()
        assert before.session.ticks == 8
        saved = before.save_checkpoint(tmp_path / "final-interruption")
        record = copy.deepcopy(before.record)
        record["tuning"]["continuation"] = saved
    finally:
        before.close()
    original = composable.policies_for

    def counted(*args, **kwargs):
        calls.append(kwargs.get("deterministic"))
        return original(*args, **kwargs)

    after = wrapper(prepared, root=tmp_path / "after-final-stop", rec=record)
    monkeypatch.setattr(composable, "policies_for", counted)
    try:
        assert run_complete(after)["done"]
        assert len(calls) == 5  # 4 final + 2 initial, less one committed mode
        assert after.session.ticks == 8
    finally:
        after.close()


def test_extra_sampling_validation_preserves_training_trajectory(tmp_path):
    prepared = dual(tmp_path / "runs")
    snapshots = []
    for index, recipe in enumerate((for_mode(prepared, "argmax"), prepared)):
        session = wrapper(recipe, finalizer=lambda _: None)
        try:
            assert run_complete(session)["done"]
            snapshots.append(
                (
                    copy.deepcopy(session.session.actions),
                    {
                        group: {
                            k: v.detach().clone()
                            for k, v in policy.network.state_dict().items()
                        }
                        for group, policy in session.session.policies.items()
                        if hasattr(policy, "network")
                    },
                )
            )
        finally:
            session.close()
    assert snapshots[0][0] == snapshots[1][0]
    for group, weights in snapshots[0][1].items():
        for name, value in weights.items():
            assert torch.equal(value, snapshots[1][1][group][name])
