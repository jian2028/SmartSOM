"""Read-only diagnostics and explicit real-backend probe routing."""

import json

import pytest

from smartsom.experiments.cli import main


def test_default_doctor_does_not_require_optional_backends_or_create_output(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "smartsom.learning.checkpoint.require_backend",
        lambda *_: pytest.fail("default doctor requested a learning backend"),
    )
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backend_probe"] == "not requested"
    assert result["output"]["root"] == str(tmp_path / "runs")
    assert not list(tmp_path.iterdir())


def test_probe_is_explicit_and_uses_selected_frozen_configuration(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda *_: {})
    monkeypatch.setattr(
        "smartsom.telemetry.training.TrainingDisplay.preflight", lambda *_: None
    )
    calls = []

    def probe(resolved, controls):
        calls.append((resolved, controls))
        return {"status": "passed", "sampled_steps": 0, "learner_updates": 0}

    monkeypatch.setattr(
        "smartsom.experiments.training_probe.probe_training_backend", probe
    )
    arguments = [
        "doctor",
        "--preset",
        "marl_micro",
        "--output-root",
        str(tmp_path / "runs"),
        "--steps",
        "128",
        "--steps-per-update",
        "64",
        "--num-envs",
        "2",
        "--sampling-processes",
        "2",
    ]
    assert main(arguments) == 0
    assert not calls and not list(tmp_path.iterdir())
    capsys.readouterr()
    assert main([*arguments, "--probe"]) == 0
    result = json.loads(capsys.readouterr().out)
    resolved, controls = calls[0]
    assert resolved.run.budget.environment_steps == 128
    assert resolved.algorithm.algorithm.provider == "rllib.resource_ppo"
    assert (controls.num_envs, controls.sampling_processes) == (2, 2)
    assert result["backend_probe"]["sampled_steps"] == 0


@pytest.mark.parametrize("options", [["--probe"], ["--device", "cuda"]])
def test_probe_or_overrides_require_a_selected_configuration(options, capsys):
    assert main(["doctor", *options]) == 2
    assert "require --preset or --config" in capsys.readouterr().err


def test_selected_cp_dependencies_are_required(monkeypatch, capsys):
    import importlib.metadata

    original = importlib.metadata.version

    def version(name):
        if name == "pyjobshop":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert main(["doctor", "--preset", "ft06_cp"]) == 2
    assert "--extra cp" in capsys.readouterr().err
