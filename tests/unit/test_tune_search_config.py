"""Search objectives must be explicit finite validation-panel scalars."""

import json
from pathlib import Path

import pytest

from smartsom.config.search import Search, validate_paths
from smartsom.experiments.tune_search import panel_objective


def spec(**changes):
    return Search.model_validate(
        {
            "algorithm": "optuna",
            "num_samples": 8,
            "seed": 42,
            "train_seed": 101,
            "objectives": [
                {
                    "panel": "sample",
                    "metric": "configured.all.delivery_fraction",
                    "mode": "max",
                }
            ],
            "parameters": {
                "algorithm.learner.lr": {
                    "distribution": "loguniform",
                    "lower": 1e-5,
                    "upper": 1e-2,
                }
            },
            **changes,
        }
    )


def test_search_does_not_optimize_seed_or_test_panel():
    with pytest.raises(ValueError, match="excluding seeds"):
        spec(
            parameters={"algorithm.seed": {"distribution": "choice", "values": [1, 2]}}
        )
    with pytest.raises(ValueError):
        spec(
            objectives=[
                {
                    "panel": "test",
                    "metric": "configured.all.delivery_fraction",
                    "mode": "max",
                }
            ]
        )


def test_unknown_parameter_rejected_before_tuner():
    with pytest.raises(ValueError, match="unknown search parameter"):
        validate_paths(spec(), {"learner": {"learning_rate": 0.01}})
    validate_paths(spec(), {"learner": {"lr": 0.01}})


@pytest.mark.parametrize("value", [True, float("inf"), "0.5"])
def test_invalid_objective_never_substitutes_penalty(tmp_path, value):
    panel = tmp_path / "sample"
    panel.mkdir()
    (panel / "summary.json").write_text(
        json.dumps({"configured": {"all": {"delivery_fraction": value}}})
    )
    with pytest.raises(ValueError, match="not finite"):
        panel_objective(tmp_path, spec().objectives[0])


def test_reads_selected_panel_metric(tmp_path):
    panel = tmp_path / "sample"
    panel.mkdir()
    (panel / "summary.json").write_text(
        json.dumps({"configured": {"all": {"delivery_fraction": 0.75}}})
    )
    assert panel_objective(tmp_path, spec().objectives[0]) == 0.75


def test_censored_objective_remains_missing(tmp_path):
    panel = tmp_path / "sample"
    panel.mkdir()
    (panel / "summary.json").write_text(
        json.dumps({"configured": {"all": {"delivery_fraction": None}}})
    )
    assert panel_objective(tmp_path, spec().objectives[0]) is None


def test_three_objectives_have_explicit_directions_and_no_priority():
    search = spec(
        objectives=[
            {"panel": "sample", "metric": "configured.all.makespan", "mode": "min"},
            {
                "panel": "sample",
                "metric": "configured.all.mean_tardiness",
                "mode": "min",
            },
            {
                "panel": "sample",
                "metric": "configured.all.final_passing_rate",
                "mode": "max",
            },
        ]
    )
    assert [objective.mode for objective in search.objectives] == ["min", "min", "max"]
    with pytest.raises(ValueError, match="distinct"):
        spec(objectives=[search.objectives[0], search.objectives[0]])


def calibrated_search(tmp_path, mode="limited"):
    """A synthetic calibration receipt bound to real compiled template inputs."""
    import yaml

    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import calibration_identity, compile_reservation
    from smartsom.experiments.reservation_runtime import identity

    repository = Path(__file__).resolve().parents[2]
    experiment = tmp_path / "experiment.yaml"
    document = {
        "schema": "smartsom.base-experiment/v1",
        "scenario": str(repository / "configs/scenarios/small.yaml"),
        "algorithm": str(repository / "configs/algorithms/base-ppo.yaml"),
        "drain_ticks": 30,
        "drain_frozen": True,
        "periodic_evaluation": {"milestones": [1]},
        "periodic_evaluation_cases": 4,
        "test_cases": 4,
        "evaluation": {"mode": mode},
        "tracking": {"mode": "disabled"},
    }
    experiment.write_text(yaml.safe_dump(document))
    initial = compile_reservation(experiment, task="train", seed=101)
    materialized = tmp_path / "calibration-inputs"
    materialized.mkdir()
    paths = load_inputs(initial.materialize(materialized))[-1]
    report = tmp_path / "original-calibration.json"
    report.write_text(
        json.dumps(
            {
                "status": "frozen",
                "calibration_identity": calibration_identity(paths),
                "inputs": identity(paths),
                "drain_ticks": 60,
                "diagnostics": {"preserve_entire_report": [1, 2, 3]},
            }
        )
    )
    document.update(drain_frozen=False, run={"calibration": str(report)})
    experiment.write_text(yaml.safe_dump(document))
    parameter = "algorithm.production.policies.ppo.learner.params.learning_rate"
    search = tmp_path / "search.yaml"
    search.write_text(
        yaml.safe_dump(
            spec(
                parameters={
                    parameter: {"distribution": "choice", "values": [0.001, 0.002]}
                }
            ).model_dump(mode="json")
        )
    )
    return experiment, search, report, parameter


@pytest.mark.parametrize("mode", ["limited", "completion"])
def test_search_candidates_share_validated_drain_and_portable_report(tmp_path, mode):
    import hashlib
    import math
    import shutil

    from smartsom.config.base_diagnostic import generate_scenario, load_inputs, seed_for
    from smartsom.config.codec import primitive
    from smartsom.experiments.reservation import allocate, arguments, load
    from smartsom.experiments.reservation_runtime import frozen_drain
    from smartsom.experiments.tune_checkpoint import restore, save
    from smartsom.experiments.tune_hpo import (
        _freeze_search,
        _read_shared_calibration,
        compile_search,
        prepare_suggestion,
    )

    experiment, search, report, parameter = calibrated_search(tmp_path, mode)
    original = report.read_bytes()
    plan = compile_search(experiment, search)
    template = allocate(plan.template, root=tmp_path / "search-run")
    _freeze_search(template, plan)

    def worlds(root):
        args, _ = arguments(root)
        ex, factory, workload, _, paths = load_inputs(root / "inputs/experiment.yaml")
        drain = frozen_drain(args, paths)
        if ex.evaluation.mode == "completion":
            drain = math.ceil(drain * ex.evaluation.completion_drain_multiplier)
        return [
            primitive(
                generate_scenario(
                    *ex.inputs_for(condition, factory, workload),
                    seed_for(prefix, 0) % (2**31),
                    condition,
                    drain,
                )
            )
            for prefix in (80000, 90000)
            for condition in ("normal", "low", "medium", "high")
        ]

    expected_worlds = worlds(template)
    report.unlink()  # Candidates depend on frozen evidence, not the external file.
    config = {"root": str(template), "search": plan.search.model_dump(mode="json")}
    for trial_id, learning_rate in (("one", 0.001), ("two", 0.002)):
        candidate = prepare_suggestion(config, {parameter: learning_rate}, trial_id)
        args, settings = arguments(candidate)
        inputs = load_inputs(candidate / "inputs/experiment.yaml")
        assert settings.calibration is None and inputs[0].drain_frozen
        assert (
            inputs[3].production.policies["ppo"].learner.params["learning_rate"]
            == learning_rate
        )
        assert frozen_drain(args, inputs[-1]) == 60
        assert (candidate / "inputs/calibration.json").read_bytes() == original
        derived = json.loads((candidate / "inputs/search-calibration.json").read_text())
        assert derived == plan.calibration
        assert derived["report_sha256"] == hashlib.sha256(original).hexdigest()
        frozen = load(candidate)
        assert "inputs/search-calibration.json" in frozen["files"]
        assert "inputs/calibration.json" in frozen["files"]
        assert "inputs/calibration-identity.json" in frozen["files"]
        assert worlds(candidate) == expected_worlds
        assert (
            prepare_suggestion(config, {parameter: learning_rate}, trial_id)
            == candidate
        )
    checkpoint = tmp_path / "portable-checkpoint"
    checkpoint.mkdir()
    save(candidate, checkpoint, {"phase": "training"})
    shutil.rmtree(template)
    portable = tmp_path / "restored-candidate"
    restore(checkpoint, portable)
    load(portable)
    metadata = json.loads((portable / "inputs/search-calibration.json").read_text())
    retained_report, identity_inputs = _read_shared_calibration(portable, metadata)
    assert retained_report == original
    assert (
        hashlib.sha256(identity_inputs).hexdigest()
        == metadata["template_calibration_identity"]
    )
    identity = json.loads(identity_inputs)
    assert identity["initial_drain_ticks"] == 30
    assert (
        identity["algorithm"]["production"]["policies"]["ppo"]["learner"]["params"][
            "learning_rate"
        ]
        == 0.0003
    )
    (portable / "inputs/calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen input identity"):
        load(portable)


@pytest.mark.parametrize(
    "change", [{"status": "incomplete"}, {"calibration_identity": "different"}]
)
def test_search_rejects_invalid_template_calibration(tmp_path, change):
    from smartsom.experiments.tune_hpo import compile_search

    experiment, search, report, _ = calibrated_search(tmp_path)
    report.write_text(json.dumps({**json.loads(report.read_text()), **change}))
    with pytest.raises(ValueError, match="matching completed template calibration"):
        compile_search(experiment, search)


def test_search_rejects_report_changed_between_check_and_freeze(tmp_path):
    from smartsom.experiments.reservation import allocate
    from smartsom.experiments.tune_hpo import _freeze_search, compile_search

    experiment, search, report, _ = calibrated_search(tmp_path)
    plan = compile_search(experiment, search)
    report.write_text(report.read_text() + "\n")
    template = allocate(plan.template, root=tmp_path / "changed-run")
    with pytest.raises(ValueError, match="changed after validation"):
        _freeze_search(template, plan)


def test_fixed_experiment_still_requires_its_own_calibration_identity(tmp_path):
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments.reservation import allocate, arguments
    from smartsom.experiments.reservation_runtime import frozen_drain

    experiment, _, _, parameter = calibrated_search(tmp_path)
    changed = compile_reservation(experiment, task="train", sets=[f"{parameter}=0.002"])
    root = allocate(changed, root=tmp_path / "fixed-run")
    args, _ = arguments(root)
    with pytest.raises(ValueError, match="matching completed calibration"):
        frozen_drain(args, load_inputs(root / "inputs/experiment.yaml")[-1])


def test_interrupted_candidate_preparation_does_not_publish_partial_plan(
    tmp_path, monkeypatch
):
    from smartsom.experiments import tune_hpo
    from smartsom.experiments.reservation import allocate, load
    from smartsom.experiments.tune_checkpoint import restore, save

    experiment, search, _, parameter = calibrated_search(tmp_path)
    plan = tune_hpo.compile_search(experiment, search)
    template = allocate(plan.template, root=tmp_path / "search-run")
    tune_hpo._freeze_search(template, plan)
    config = {"root": str(template), "search": plan.search.model_dump(mode="json")}
    candidate = template / "trials/one"
    candidate.mkdir(parents=True)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    save(candidate, bootstrap, {"phase": "training"})
    original = tune_hpo._bind_files

    def interrupted(root, names):
        assert (root / "inputs/calibration.json").is_file()
        raise RuntimeError("interrupted before complete candidate publication")

    monkeypatch.setattr(tune_hpo, "_bind_files", interrupted)
    with pytest.raises(RuntimeError, match="before complete candidate"):
        tune_hpo.prepare_suggestion(config, {parameter: 0.002}, "one")
    assert not (candidate / "plan.json").exists()
    assert not list((template / "control").glob("suggestion-*"))
    assert restore(bootstrap, candidate) == {"phase": "training"}
    monkeypatch.setattr(tune_hpo, "_bind_files", original)
    assert tune_hpo.prepare_suggestion(config, {parameter: 0.002}, "one") == candidate
    frozen = load(candidate)
    assert "suggestion.json" in frozen["files"]
    assert "inputs/calibration.json" in frozen["files"]
