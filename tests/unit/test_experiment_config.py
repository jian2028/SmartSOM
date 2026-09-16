"""Public recipes preserve scientific identity across entry points and locations."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from smartsom.api import load_config, load_preset, show_config
from smartsom.config.codec import ConfigurationError, canonical_json, primitive
from smartsom.config.experiment import (
    PRESETS,
    apply_overrides,
    from_legacy,
    prepare,
    prepare_frozen,
)
from smartsom.config.training import resolve_training_run
from smartsom.experiments.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_cli_yaml_and_python_resolve_to_same_science(tmp_path, capsys):
    python = load_preset("marl_micro")
    python.runtime.num_envs = 4
    python.algorithm.learning_rate = 0.0007
    python.output.name = "python"
    cli = apply_overrides(
        load_preset("marl_micro"),
        [
            ("runtime.num_envs", 4),
            ("algorithm.learning_rate", 0.0007),
            ("output.name", "cli"),
        ],
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "preset: marl_micro\nruntime:\n  num_envs: 4\nalgorithm:\n  learning_rate: 0.0007\noutput:\n  name: yaml\n"
    )
    file = load_config(path)
    results = [prepare(c) for c in (python, cli, file)]
    assert len({r.scientific_sha256 for r in results}) == 1
    assert python.origins()["algorithm.learning_rate"] == "python"
    assert cli.origins()["algorithm.learning_rate"] == "cli"
    assert file.origins()["algorithm.learning_rate"] == str(path)
    assert (
        main(
            [
                "show-config",
                "--preset",
                "marl_micro",
                "--num-envs",
                "4",
                "--set",
                "algorithm.learning_rate=0.0007",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["scientific_sha256"]
        == results[0].scientific_sha256
    )


@pytest.mark.parametrize("name", PRESETS)
def test_preview_is_read_only_and_does_not_load_backends(name, tmp_path, monkeypatch):
    import smartsom.engine.production as engine
    import smartsom.learning.checkpoint as checkpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        engine, "ProductionSimulator", lambda *a, **kw: pytest.fail("preview simulated")
    )
    monkeypatch.setattr(
        checkpoint,
        "require_backend",
        lambda *a: pytest.fail("preview required a backend"),
    )
    if name == "ft06_cp":
        with pytest.raises(ConfigurationError, match="CP-SAT has no grid"):
            show_config(load_preset(name))
        assert not list(tmp_path.iterdir())
        return
    details = show_config(load_preset(name))
    assert details["inputs"]["jobs"] > 0
    assert not list(tmp_path.iterdir())


def test_logging_and_origins_do_not_change_scientific_inputs():
    config = load_preset("marl_micro")
    before = prepare(config)
    config.logging.verbose = False
    config.logging.progress = "off"
    config.logging.tensorboard = False
    config.output.tags = ("demo",)
    assert prepare(config).scientific_sha256 == before.scientific_sha256
    config.algorithm.gamma = 0.99
    assert prepare(config).scientific_sha256 != before.scientific_sha256


def test_snapshot_detaches_mutable_configuration():
    config = load_preset("marl_micro")
    frozen = prepare(config)
    config.algorithm.learning_rate = 0.009
    assert json.loads(frozen.config_json)["algorithm"]["learning_rate"] == 0.0003
    assert frozen.resolved.algorithm.learning_rate == 0.0003


def test_public_extensions_bind_versions_and_allow_parameter_search():
    from smartsom.config.extensions import (
        ExtensionRef,
        ExtensionSpec,
        NetworkSpec,
        RewardSpec,
    )

    config = load_preset("sb3_micro")
    config.algorithm.extensions = ExtensionSpec(
        network=NetworkSpec(),
        reward=RewardSpec(
            team=ExtensionRef(
                name="builtin.reward_scale", version="1", parameters={"scale": 0.5}
            )
        ),
    )
    frozen = prepare(config)
    selected = frozen.resolved.algorithm.extensions
    assert selected.network.actor.encoder.code_sha256
    assert selected.reward.team.code_sha256
    changed = apply_overrides(
        config, [("algorithm.extensions.network.actor.hidden_sizes", [32, 16])]
    )
    assert prepare(
        changed
    ).resolved.algorithm.extensions.network.actor.hidden_sizes == (32, 16)
    assert prepare(changed).scientific_sha256 != frozen.scientific_sha256


def test_optional_extension_fields_can_be_authored_by_strict_cli_overrides():
    from smartsom.config.extensions import ExtensionSpec, NetworkBranch, NetworkSpec

    config = load_preset("marl_micro")
    changed = apply_overrides(
        config,
        [
            ("algorithm.extensions.network.actor.hidden_sizes", [32, 16]),
            ("algorithm.extensions.network.actor.activation", "relu"),
        ],
    )
    config.algorithm.extensions = ExtensionSpec(
        network=NetworkSpec(
            actor=NetworkBranch(hidden_sizes=(32, 16), activation="relu")
        )
    )
    assert prepare(changed).scientific_sha256 == prepare(config).scientific_sha256
    for path, value in (
        ("algorithm.extensions.network.actor.hidden_size", [32]),
        ("algorithm.extensions.network.roles.unknown.actor.hidden_sizes", [32]),
        ("algorithm.extensions.network.actor.hidden_sizes", ["32"]),
        ("algorithm.extensions.network.actor..activation", "relu"),
    ):
        with pytest.raises(ConfigurationError):
            apply_overrides(load_preset("marl_micro"), [(path, value)])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("training.total_steps", 4097, "whole updates"),
        ("runtime.num_envs", 3, "divide evenly"),
        ("algorithm.batch_size", 63, "whole.*minibatches"),
        ("runtime.sampling_processes", 2, "cannot exceed"),
        ("training.total_steps", "4096", "valid integer"),
        ("runtime.device", "mps", "cpu.*cuda"),
        ("algorithm.learning_rate", float("nan"), "JSON compliant"),
        ("evaluation.seed", 303, "distinct"),
    ],
)
def test_invalid_fields_fail_before_execution(field, value, match):
    with pytest.raises((ConfigurationError, ValueError), match=match):
        apply_overrides(load_preset("marl_micro"), [(field, value)])


def test_direct_assignment_and_execution_boundary_validation():
    config = load_preset("marl_micro")
    with pytest.raises(ValidationError, match="valid integer"):
        config.training.total_steps = "4096"
    config.training.total_steps = 4097
    with pytest.raises(ConfigurationError, match="whole updates"):
        prepare(config)


@pytest.mark.parametrize(
    "args",
    [
        ["--steps", "4096", "--steps", "4096"],
        ["--steps", "4096", "--set", "training.total_steps=4096"],
        ["--set", "training.total_steps=4096", "--set", "training={total_steps: 4096}"],
        ["--set", "algorithm.lerning_rate=0.01"],
    ],
)
def test_cli_rejects_duplicate_overlapping_and_unknown_overrides(args, capsys):
    assert main(["show-config", "--preset", "marl_micro", *args]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_yaml_error_reports_file_and_field(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("preset: marl_micro\nalgorithm:\n  learning_rate: wrong\n")
    with pytest.raises(
        ConfigurationError, match="invalid.yaml.*algorithm.learning_rate"
    ):
        try:
            load_config(path)
        except ConfigurationError as exc:
            raise ConfigurationError(str(exc).replace("\n", " ")) from exc
    path.write_text("preset: marl_micro\nseed: 1\nseed: 2\n")
    with pytest.raises(ConfigurationError, match="duplicate key"):
        load_config(path)


@pytest.mark.parametrize("name", ["learning_marl", "learning_rllib", "learning_sb3"])
def test_migration_preserves_frozen_training_recipe(name, tmp_path, monkeypatch):
    import smartsom.learning.checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "require_backend", lambda *a: None)
    path = ROOT / "configs/runs" / f"{name}.yaml"
    original = resolve_training_run(path).resolved
    converted = from_legacy(path)
    target = tmp_path / "converted.json"
    target.write_text(canonical_json(converted))
    actual = prepare(load_config(target)).resolved
    assert actual.algorithm == original.algorithm
    assert actual.training_json == original.training_json
    assert actual.episode(0) == original.episode(0)
    assert not converted.validation.enabled
    assert not converted.logging.tensorboard


def test_cp_migration_preserves_explicit_solver_budget(tmp_path):
    config = from_legacy(ROOT / "configs/runs/ft06_cp.yaml")
    config.solver.time_limit_seconds = 17.5
    assert primitive(config)["solver"]["time_limit_seconds"] == 17.5
    with pytest.raises(ConfigurationError, match="CP-SAT has no grid"):
        prepare(config, training=False)


def test_declared_paths_are_relative_to_user_file(tmp_path):
    from smartsom.config.authoring import create_template

    project = create_template("minimal_jsp", tmp_path / "project")
    recipe = project / "experiment.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "preset": "test",
                "scenario": "scenario.yaml",
                "algorithm": {"source": "algorithm.yaml"},
                "scenario_overrides": {
                    "factory": "factory.yaml",
                    "workload": "workload.yaml",
                },
                "output": {"root": "output"},
            }
        )
    )
    config = load_config(recipe)
    assert config.output.root == str(project / "output")
    assert config.scenario_overrides["factory"] == str(project / "factory.yaml")
    assert show_config(config)["inputs"]["jobs"] > 0


def test_scenario_overrides_retain_contract_validation():
    config = apply_overrides(
        load_preset("marl_micro"),
        [
            ("scenario_overrides.arrivals.initial_jobs", 1),
            ("scenario_overrides.mode", "dynamic"),
        ],
    )
    assert (
        json.loads(prepare(config).resolved.settings_json)["arrivals"]["initial_jobs"]
        == 1
    )
    bad = apply_overrides(config, [("scenario_overrides.mode", "static")])
    with pytest.raises(ValueError, match="generated arrivals require dynamic"):
        prepare(bad)


def test_no_preview_mutation_of_public_object():
    config = load_preset("marl_micro")
    before = primitive(config)
    preview = show_config(config)
    preview["config"]["algorithm"]["learning_rate"] = 1.0
    assert primitive(config) == before


def test_frozen_candidate_uses_same_parameter_binding_and_world():
    template = prepare(load_preset("marl_micro"))
    candidate = apply_overrides(
        load_preset("marl_micro"),
        [
            ("algorithm.learning_rate", 0.0007),
            ("training.total_steps", 512),
        ],
    )
    frozen = prepare_frozen(candidate, template)
    resolved = prepare(candidate)
    assert frozen.scientific_sha256 == resolved.scientific_sha256
    assert frozen.resolved.episode(2) == resolved.resolved.episode(2)
    candidate.seed = 102
    with pytest.raises(ConfigurationError, match="retain seed, scenario"):
        prepare_frozen(candidate, template)


def test_bundled_learning_presets_keep_original_workload_and_explicit_grid():
    from smartsom.config.codec import digest

    workspace = prepare(load_config(ROOT / "configs/runs/learning_marl.yaml")).resolved
    for name in ("marl_micro", "rllib_micro", "sb3_micro"):
        recipe = prepare(load_preset(name)).resolved
        assert digest(recipe.scenario.factory) == digest(workspace.scenario.factory)
        assert recipe.scenario.demands == workspace.scenario.demands
        assert len(recipe.scenario.factory.machines) == 8
        assert len(recipe.scenario.factory.agvs) == 4
        assert sum(len(d.steps) for d in recipe.scenario.demands) == 9
