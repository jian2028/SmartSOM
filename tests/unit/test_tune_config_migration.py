"""Removed scheduler controls fail with actionable, explicit migration errors."""

from pathlib import Path

import pytest

from smartsom.config.base_diagnostic import ReservationBatch, ReservationRun
from smartsom.config.launch import Concurrency, Resources
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.cli import _parser

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "field", ["tune", "resource_execution", "calibration_seconds", "probe_ticks"]
)
def test_deleted_run_fields_are_not_silently_ignored(field):
    with pytest.raises(ValueError, match="removed run fields"):
        ReservationRun.model_validate({field: 1})


def test_scientific_calibration_and_unbounded_training_remain():
    run = ReservationRun(task="train", seconds=None, calibration="frozen-drain.json")
    assert run.calibration == "frozen-drain.json"
    assert run.max_failures == 0
    assert Concurrency().max_concurrent is None


@pytest.mark.parametrize(
    "model,values,message",
    [
        (Concurrency, {"mode": "manual"}, "concurrency.mode was removed"),
        (Resources, {"guard": None}, "resources.guard was removed"),
        (ReservationBatch, {"parallel_files": 1}, "batch.parallel_files was removed"),
    ],
)
def test_deleted_operational_controls_explain_migration(model, values, message):
    with pytest.raises(ValueError, match=message):
        model.model_validate(values)


@pytest.mark.parametrize("command", ["add", "close", "remove", "configure"])
def test_online_commands_no_longer_exposed(command):
    with pytest.raises(SystemExit):
        _parser().parse_args([command, "unused"])


def test_static_profile_rejects_impossible_phase(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "resources:\n  cpus: 2\ntrial_resources:\n  training:\n    cpus: 3\n"
    )
    with pytest.raises(ValueError, match="exceeding total"):
        compile_reservation(
            ROOT / "experiments/examples/small/default/experiment.yaml", profile=profile
        )


@pytest.mark.parametrize(
    "evidence",
    [
        "episodes.jsonl",
        "progress.json",
        "continuation/current.json",
        "trials/old/receipt.json",
    ],
)
def test_missing_tune_state_cannot_restart_existing_science(tmp_path, evidence):
    from smartsom.experiments.tune_execution import restore_required

    root = tmp_path / "run"
    path = root / evidence
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="no Tune experiment checkpoint"):
        restore_required(root / "ray/search", [root], checkpoint_available=False)
    assert path.read_bytes() == before


def test_tune_restore_selection_rejects_incomplete_controller_and_explicit_overwrite(
    tmp_path,
):
    from smartsom.experiments.tune_execution import restore_required

    root = tmp_path / "run"
    storage = root / "ray/search"
    assert not restore_required(storage, [root], checkpoint_available=False)
    assert restore_required(storage, [root], checkpoint_available=True)
    with pytest.raises(ValueError, match="already exists"):
        restore_required(storage, [root], checkpoint_available=True, requested=False)
    with pytest.raises(ValueError, match="checkpoint is missing"):
        restore_required(storage, [root], checkpoint_available=False, requested=True)
    storage.mkdir(parents=True)
    (storage / "smartsom-execution.json").write_text("{}")
    with pytest.raises(ValueError, match="refusing a fresh run"):
        restore_required(storage, [root], checkpoint_available=False)


def test_authored_budget_requires_profile_but_frozen_snapshot_recompiles(tmp_path):
    import yaml

    from tests.unit.test_algorithm_components import author, recipe

    path = author(tmp_path, recipe())
    value = yaml.safe_load(path.read_text())
    value["resources"] = {"cpus": 2}
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="resources belongs in an execution profile"):
        compile_reservation(path)
    value.pop("resources")
    path.write_text(yaml.safe_dump(value))
    profile = tmp_path / "profile.yaml"
    profile.write_text("resources:\n  cpus: 2\n")
    compiled = compile_reservation(path, profile=profile)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    frozen = compiled.materialize(inputs)
    restored = compile_reservation(frozen)
    assert restored.operations["resources"]["cpus"] == 2
    assert restored.scientific_sha256 == compiled.scientific_sha256


def test_failed_search_retry_is_rejected_before_loading_or_launching(
    tmp_path, monkeypatch
):
    from smartsom.experiments import tune_hpo
    from smartsom.experiments.tune_execution import fit_cohort

    sentinel = tmp_path / "failed-search.json"
    sentinel.write_text("retain failure evidence")
    monkeypatch.setattr(
        tune_hpo, "bind", lambda *_: pytest.fail("retry bound mutable display")
    )
    with pytest.raises(ValueError, match="cannot be retried in place"):
        tune_hpo.execute(tmp_path, retry_failed=True)
    with pytest.raises(ValueError, match="cannot be retried in place"):
        fit_cohort(
            [],
            storage=tmp_path / "ray",
            resources={},
            search=object(),
            retry_failed=True,
        )
    assert sentinel.read_text() == "retain failure evidence"
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize("task", ["smoke", "calibrate", "train"])
@pytest.mark.parametrize(
    "setting",
    [
        "experiment.replay.train=all",
        "experiment.evaluation.window_analysis={throughput_width_ticks: 10, throughput_step_ticks: 5, recovery_window_ticks: 10}",
        "experiment.evaluation.common_window_end_tick=20",
    ],
)
def test_public_check_rejects_deferred_features_without_allocating(
    tmp_path, task, setting
):
    from smartsom import api

    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="deferred"):
        api.check_experiment(
            ROOT / "experiments/examples/current-smoke/default/experiment.yaml",
            task=task,
            sets=[
                setting,
                "experiment.drain_frozen=true",
                f'experiment.run.output="{output}"',
            ],
        )
    assert not output.exists()


def test_public_run_rejects_native_learner_before_allocating(tmp_path):
    from smartsom import api

    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="migrate learner"):
        api.run_experiment(
            ROOT / "experiments/examples/small/default/experiment.yaml",
            task="train",
            sets=["experiment.drain_frozen=true", f'experiment.run.output="{output}"'],
        )
    assert not output.exists()


def test_public_fixed_rule_panel_uses_scientific_resources(tmp_path):
    from smartsom import api

    source = ROOT / "experiments/current/small/rule/experiment.yaml"
    output = tmp_path / "must-not-exist"
    api.check_experiment(
        source,
        sets=[f'experiment.run.output="{output}"'],
        operations={
            "resources": {"cpus": 3},
            "rllib": {"evaluation_num_env_runners": 2},
        },
    )
    assert not output.exists()
    with pytest.raises(ValueError, match="exceeding"):
        api.check_experiment(
            source,
            operations={
                "resources": {"cpus": 2},
                "rllib": {"evaluation_num_env_runners": 2},
            },
        )
    with pytest.raises(ValueError, match="deferred"):
        api.check_experiment(source, sets=["experiment.run.methods=[configured]"])


@pytest.mark.parametrize(
    "setting,message",
    [
        ("num_atoms=51", "num_atoms=1"),
        ("noisy=true", "noisy=False"),
        ("store_buffer_in_checkpoints=false", "store_buffer_in_checkpoints=True"),
        (
            "replay_buffer_config={type: UnsupportedReplay}",
            "approved restorable official replay",
        ),
    ],
)
def test_public_dqn_rejects_unsupported_training_before_allocation(
    tmp_path, setting, message
):
    import yaml

    from smartsom import api

    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.base-experiment/v1",
                "scenario": str(ROOT / "configs/scenarios/small.yaml"),
                "algorithm": str(ROOT / "configs/algorithms/rllib-base-dqn.yaml"),
                "drain_frozen": True,
                "train_ticks": 1,
                "train_seeds": [101],
                "periodic_evaluation_cases": 4,
                "test_cases": 4,
            }
        )
    )
    output = tmp_path / "must-not-exist"
    for entry in (api.check_experiment, api.run_experiment):
        with pytest.raises(ValueError, match=message):
            entry(
                path,
                task="train",
                sets=[
                    f"algorithm.production.policies.dqn.learner.params.{setting}",
                    f'experiment.run.output="{output}"',
                ],
            )
        assert not output.exists()
