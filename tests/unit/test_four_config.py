"""Four author files compile into detached native plans without execution."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from smartsom.config.authoring_v4 import AlgorithmV2, ExperimentV4
from smartsom.config.codec import ConfigurationError, primitive
from smartsom.config.experiment_v4 import compile_experiment


def test_cli_positional_check_and_config_view_are_read_only(author_files, capsys):
    from smartsom.experiments.cli import main

    paths, _ = author_files
    for command in ("check", "show-config", "validate"):
        assert main([command, str(paths["experiment"])]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["task"] == "evaluate" and payload["count"] == 1
        assert payload["input_type"] == "experiment-v4"
    assert not (paths["experiment"].parent.parent / "results").exists()


def test_cli_run_seed_settings_and_preview_reach_frozen_plan(
    author_files, monkeypatch, capsys
):
    from smartsom.experiments.cli import main

    paths, _ = author_files
    plans = []
    monkeypatch.setattr(
        "smartsom.experiments.author_driver.run",
        lambda plan, **kwargs: plans.append(plan) or {"status": "completed"},
    )
    assert (
        main(
            [
                "run",
                str(paths["experiment"]),
                "--seed",
                "505",
                "--data-seed",
                "909",
                "--preview",
                "--set",
                "experiment.output.name=override",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    plan = plans[0]
    assert plan.experiment.seed == 505 and plan.experiment.data_seed == 909
    assert plan.experiment.output.name == "override"
    prepared = plan.entries[0].prepared
    assert len(json.loads(prepared.evaluation_json)) == 1
    assert (
        prepared.config.evaluation.record and not prepared.config.evaluation.full_replay
    )
    assert (
        json.loads(prepared.training_inputs_json)["authoring"]["purpose"] == "preview"
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--preview", "--set", "experiment.evaluation.replications=2"],
        ["--seed", "10", "--set", "experiment.seed=11"],
        ["--background", "--set", "experiment.execution.background=false"],
        ["--performance", "auto"],
        ["--sampling-processes", "2"],
    ],
)
def test_cli_rejects_conflicting_or_inapplicable_author_flags(
    author_files, flags, capsys
):
    from smartsom.experiments.cli import main

    paths, _ = author_files
    assert main(["check", str(paths["experiment"]), *flags]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_v4_data_seed_is_the_only_author_data_root(author_files):
    paths, documents = author_files
    documents["experiment"]["evaluation"]["seed"] = 9
    write(paths["experiment"], documents["experiment"])
    with pytest.raises(ConfigurationError, match="data_seed"):
        compile_experiment(paths["experiment"])


@pytest.fixture
def author_files(tmp_path):
    from smartsom.config.factory_design import load_factory_design_file

    source = (
        Path(__file__).resolve().parents[2] / "configs/test/factories/factory_hand.yaml"
    )
    # This test is also used from an isolated test checkout against current source.
    if not source.exists():
        source = Path(
            "/Users/jianni/code/SmartSOM/configs/test/factories/factory_hand.yaml"
        )
    factory, _ = load_factory_design_file(source)
    directory = tmp_path / "author"
    directory.mkdir()
    inputs = directory / "inputs"
    inputs.mkdir()
    documents = {
        "factory": primitive(factory),
        "workload": {
            "schema": "smartsom.workload/v2",
            "demands": [
                {
                    "demand_id": "job-1",
                    "steps": [
                        {
                            "operation_id": "op-1",
                            "operation_type": "operation_1",
                            "nominal_ticks": 2,
                        }
                    ],
                    "release_at": 0,
                    "due_at": 30,
                }
            ],
        },
        "algorithm": {
            "schema": "smartsom.algorithm/v2",
            "mode": "rules",
            "agents": {
                role: {"default": {"kind": "rule", "name": name}}
                for role, name in {
                    "machine": "spt",
                    "buffer": "edd",
                    "dispatcher": "nearest",
                    "mover": "shortest_path",
                }.items()
            },
        },
        "experiment": {
            "schema": "smartsom.experiment-config/v4",
            "task": "evaluate",
            "factory": "inputs/factory.yaml",
            "workload": "inputs/workload.yaml",
            "algorithm": "inputs/algorithm.yaml",
            "data_seed": 9001,
            "evaluation": {"replications": 1, "full_replay": False},
            "output": {"root": "../results", "name": "tiny-check"},
            "logging": {"progress": "off", "verbose": False},
        },
    }
    paths = {}
    for kind, data in documents.items():
        path = (
            directory / "experiment.yaml"
            if kind == "experiment"
            else inputs / f"{kind}.yaml"
        )
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        paths[kind] = path
    return paths, documents


def write(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def central_algorithm():
    return {
        "schema": "smartsom.algorithm/v2",
        "mode": "central",
        "learner": {"backend": "sb3", "algorithm": "ppo"},
        "controller": {"kind": "new_model"},
    }


def training_experiment(documents, task="train"):
    data = copy.deepcopy(documents["experiment"])
    data["task"] = task
    data["training"] = {"total_ticks": 64, "ticks_per_update": 32, "max_ticks": 64}
    data["validation"] = {"enabled": False}
    data["evaluation"]["checkpoint"] = "last"
    return data


def test_check_freezes_four_sources_without_allocating_output_or_mutating_files(
    author_files, tmp_path
):
    paths, documents = author_files
    before = {p: p.read_bytes() for p in paths.values()}
    existing = set(tmp_path.rglob("*"))
    plan = compile_experiment(paths["experiment"])
    summary = plan.summary()
    assert summary["status"] == "checked" and summary["count"] == 1
    assert summary["task"] == "evaluate"
    assert summary["execution"]["executor"] == "native"
    assert summary["execution"]["performance"] == "off"
    assert summary["execution"]["max_concurrent"] == 1
    assert not summary["execution"]["background"]
    assert Path(summary["output"]["root"]) == tmp_path / "results"
    assert not (tmp_path / "results").exists()
    assert set(tmp_path.rglob("*")) == existing
    assert {p: p.read_bytes() for p in paths.values()} == before
    entry = plan.entries[0]
    assert (
        entry.prepared.config.scenario is None
        and entry.prepared.config.composition is None
    )
    assert set(json.loads(entry.prepared.policies_json)) == set(
        documents["algorithm"]["agents"]
    )
    assert set(entry.sources) == {"factory", "workload", "algorithm", "experiment"}
    for key, path in paths.items():
        assert entry.sources[key] == str(path.resolve())
    frozen = json.loads(entry.prepared.training_inputs_json)["authoring"]
    assert frozen["documents"]["algorithm"]["mode"] == "rules"
    assert set(frozen["input_sha256"]) == {str(p.resolve()) for p in paths.values()}
    assert len(json.loads(entry.prepared.evaluation_json)) == 1
    assert not json.loads(entry.prepared.validation_json)
    assert entry.prepared.config.training is None


def test_paths_resolve_against_experiment_and_selected_algorithm_file(
    author_files, monkeypatch, tmp_path
):
    paths, _ = author_files
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    plan = compile_experiment(paths["experiment"])
    assert plan.entries[0].sources["factory"] == str(paths["factory"])
    replacement = elsewhere / "rules.yaml"
    replacement.write_bytes(paths["algorithm"].read_bytes())
    selected = compile_experiment(paths["experiment"], algorithm=replacement)
    assert selected.entries[0].sources["algorithm"] == str(replacement)


def test_selectors_precede_typed_namespace_overrides_and_preserve_originals(
    author_files, tmp_path
):
    paths, documents = author_files
    alternate = tmp_path / "alternate-factory.yaml"
    factory = copy.deepcopy(documents["factory"])
    factory["factory"]["name"] = "Selected factory"
    write(alternate, factory)
    original = alternate.read_bytes()
    plan = compile_experiment(
        paths["experiment"],
        factory=alternate,
        sets=(
            "factory.factory.name=Overridden factory",
            "experiment.execution.max_concurrent=2",
            "experiment.evaluation.replications=2",
            "experiment.runtime.environment.tick_limit=45",
        ),
    )
    entry = plan.entries[0]
    assert entry.sources["factory"] == str(alternate)
    assert entry.prepared.scenario.factory.name == "Overridden factory"
    assert entry.prepared.scenario.tick_limit == 45
    assert plan.experiment.execution.max_concurrent == 2
    assert len(json.loads(entry.prepared.evaluation_json)) == 2
    assert alternate.read_bytes() == original
    assert "Overridden factory" not in paths["factory"].read_text()


@pytest.mark.parametrize(
    "override",
    [
        "experiment.runtime.unknown=1",
        "factory.factory.grid.width=true",
        "workload.future.hidden=true",
        "algorithm.social=true",
        "unknown.field=1",
        "experiment.training.total_ticks=3",
        "experiment.runtime.num_envs=0",
    ],
)
def test_unknown_or_wrongly_typed_overrides_rejected(author_files, override):
    paths, _ = author_files
    with pytest.raises(ConfigurationError):
        compile_experiment(paths["experiment"], sets=(override,))


def test_overlapping_or_duplicate_overrides_rejected(author_files):
    paths, _ = author_files
    for items in (
        (
            "experiment.execution.max_concurrent=2",
            "experiment.execution.max_concurrent=3",
        ),
        (
            "experiment.execution={max_concurrent: 2}",
            "experiment.execution.max_concurrent=3",
        ),
    ):
        with pytest.raises(ConfigurationError, match="duplicate/overlapping"):
            compile_experiment(paths["experiment"], sets=items)


def test_default_optimizer_fields_can_be_overridden_without_parameter_boilerplate(
    author_files,
):
    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    write(paths["experiment"], training_experiment(documents))
    plan = compile_experiment(
        paths["experiment"],
        sets=(
            "algorithm.learner.parameters.learning_rate=0.0001",
            "experiment.training.total_ticks=96",
        ),
    )
    entry = plan.entries[0]
    assert entry.prepared.config.training.total_ticks == 96
    assert entry.prepared.config.training.backend == "sb3"
    assert entry.prepared.config.training.mode == "central"
    assert json.loads(entry.prepared.parameters_json)["learning_rate"] == 0.0001
    assert json.loads(entry.prepared.parameters_json)["n_epochs"] == 4


@pytest.mark.parametrize(
    "change",
    [
        "central_agents",
        "central_no_controller",
        "central_no_backend",
        "central_dqn",
        "resource_sb3",
        "rules_learner",
        "rules_model",
        "missing_role",
        "unknown_role",
    ],
)
def test_invalid_algorithm_modes_are_rejected(author_files, change):
    _, documents = author_files
    data = (
        central_algorithm()
        if change.startswith("central")
        else copy.deepcopy(documents["algorithm"])
    )
    if change == "central_agents":
        data["agents"] = documents["algorithm"]["agents"]
    elif change == "central_no_controller":
        del data["controller"]
    elif change == "central_no_backend":
        del data["learner"]["backend"]
    elif change == "central_dqn":
        data["learner"]["algorithm"] = "dqn"
    elif change == "resource_sb3":
        data["mode"] = "resource"
        data["learner"] = {"algorithm": "ppo", "backend": "sb3"}
    elif change == "rules_learner":
        data["learner"] = {"algorithm": "ppo", "backend": "rllib"}
    elif change == "rules_model":
        data["agents"]["machine"]["default"] = {"kind": "new_model"}
    elif change == "missing_role":
        del data["agents"]["mover"]
    else:
        data["agents"]["quality"] = data["agents"]["machine"]
    with pytest.raises(ValueError):
        AlgorithmV2.model_validate_json(json.dumps(data))


def test_rules_training_and_unsupported_performance_modes_rejected(author_files):
    paths, documents = author_files
    for task in ("train", "train-evaluate"):
        write(paths["experiment"], training_experiment(documents, task))
        with pytest.raises(
            ConfigurationError, match="rules mode supports evaluate only"
        ):
            compile_experiment(paths["experiment"])
    write(paths["experiment"], documents["experiment"])
    for performance in ("recommend", "auto"):
        with pytest.raises(ConfigurationError, match="performance/Tune"):
            compile_experiment(paths["experiment"], performance=performance)


def test_matrix_expands_factory_workload_cartesian_product_and_selectors_narrow_axes(
    author_files,
):
    paths, documents = author_files
    second_factory = paths["factory"].with_name("factory-second.yaml")
    second_workload = paths["workload"].with_name("workload-second.yaml")
    factory = copy.deepcopy(documents["factory"])
    factory["factory"]["factory_id"] = "factory-second"
    write(second_factory, factory)
    workload = copy.deepcopy(documents["workload"])
    workload["demands"][0]["due_at"] = 40
    write(second_workload, workload)
    experiment = copy.deepcopy(documents["experiment"])
    del experiment["factory"], experiment["workload"]
    experiment["matrix"] = {
        "factories": ["inputs/factory.yaml", "inputs/factory-second.yaml"],
        "workloads": ["inputs/workload.yaml", "inputs/workload-second.yaml"],
    }
    write(paths["experiment"], experiment)
    plan = compile_experiment(paths["experiment"])
    assert len(plan.entries) == 4
    assert {(e.sources["factory"], e.sources["workload"]) for e in plan.entries} == {
        (str(f), str(w))
        for f in (paths["factory"], second_factory)
        for w in (paths["workload"], second_workload)
    }
    selected = compile_experiment(
        paths["experiment"], factory=second_factory, workload=second_workload
    )
    assert len(selected.entries) == 1
    assert selected.entries[0].sources["factory"] == str(second_factory)
    assert selected.entries[0].sources["workload"] == str(second_workload)


def test_duplicate_combinations_reject_different_paths_to_identical_inputs(
    author_files,
):
    paths, documents = author_files
    duplicate = paths["workload"].with_name("duplicate.yaml")
    duplicate.write_bytes(paths["workload"].read_bytes())
    experiment = copy.deepcopy(documents["experiment"])
    del experiment["factory"], experiment["workload"]
    experiment["matrix"] = {
        "factories": ["inputs/factory.yaml"],
        "workloads": ["inputs/workload.yaml", "inputs/duplicate.yaml"],
    }
    write(paths["experiment"], experiment)
    with pytest.raises(ConfigurationError, match="duplicate execution combination"):
        compile_experiment(paths["experiment"])


def test_breakdown_on_off_are_distinct_factory_conditions(author_files):
    paths, documents = author_files
    broken = paths["factory"].with_name("with-breakdown.yaml")
    factory = copy.deepcopy(documents["factory"])
    factory["reliability"] = {
        "enabled": True,
        "defaults": {
            "uptime": {"distribution": "uniform", "min_ticks": 4, "max_ticks": 4},
            "repair": {"min_ticks": 2, "max_ticks": 2},
        },
    }
    write(broken, factory)
    experiment = copy.deepcopy(documents["experiment"])
    del experiment["factory"], experiment["workload"]
    experiment["matrix"] = {
        "factories": ["inputs/factory.yaml", "inputs/with-breakdown.yaml"],
        "workloads": ["inputs/workload.yaml"],
    }
    write(paths["experiment"], experiment)
    plan = compile_experiment(paths["experiment"])
    assert len(plan.entries) == 2
    assert not plan.entries[0].prepared.scenario.outages
    assert plan.entries[1].prepared.scenario.outages
    assert (
        plan.entries[0].prepared.scientific_sha256
        != plan.entries[1].prepared.scientific_sha256
    )


def test_data_randomness_is_independent_of_policy_learning_seed(author_files):
    paths, documents = author_files
    workload = {
        "schema": "smartsom.workload/v2",
        "profile": {
            "jobs": 8,
            "route": ["operation_1", "operation_3"],
            "nominal_min": 2,
            "nominal_max": 20,
            "due_at": 200,
        },
    }
    write(paths["workload"], workload)
    first = compile_experiment(paths["experiment"], seed=101, data_seed=9001).entries[0]
    second = compile_experiment(paths["experiment"], seed=202, data_seed=9001).entries[
        0
    ]
    other_data = compile_experiment(
        paths["experiment"], seed=101, data_seed=9002
    ).entries[0]
    assert first.prepared.scenario.demands == second.prepared.scenario.demands
    assert first.prepared.scenario_json == second.prepared.scenario_json
    assert first.workload == second.workload
    assert first.prepared.scenario.demands != other_data.prepared.scenario.demands
    assert first.prepared.scientific_sha256 != second.prepared.scientific_sha256


def test_evaluation_cannot_use_training_seed_axis(author_files):
    _, documents = author_files
    experiment = copy.deepcopy(documents["experiment"])
    del experiment["factory"], experiment["workload"]
    experiment["matrix"] = {
        "factories": ["f.yaml"],
        "workloads": ["w.yaml"],
        "seeds": [1, 2],
    }
    with pytest.raises(ValueError, match="training-seed axis"):
        ExperimentV4.model_validate_json(json.dumps(experiment))


def test_rules_check_does_not_import_optional_frameworks(author_files):
    paths, _ = author_files
    code = r"""
import builtins, json, sys
original = builtins.__import__
blocked = {"ray", "torch", "gymnasium", "stable_baselines3", "sb3_contrib"}
def checked(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("check imported optional framework: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from smartsom.config.experiment_v4 import compile_experiment
plan = compile_experiment(sys.argv[1])
assert len(plan.entries) == 1
assert not blocked.intersection(sys.modules)
print(json.dumps(plan.summary()))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(
        Path(sys.modules["smartsom"].__file__).resolve().parent.parent
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(paths["experiment"])],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["task"] == "evaluate"


@pytest.mark.parametrize("task", ["train", "train-evaluate"])
def test_central_learning_tasks_compile_without_starting_frameworks(
    author_files, task, tmp_path
):
    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    write(paths["experiment"], training_experiment(documents, task))
    plan = compile_experiment(paths["experiment"])
    assert len(plan.entries) == 1 and plan.entries[0].task == task
    assert plan.entries[0].prepared.config.training.groups == ("central",)
    assert plan.entries[0].prepared.config.training.total_ticks == 64
    assert not (tmp_path / "results").exists()
    summary = plan.summary()["entries"][0]
    assert summary["evaluation_cases"] == (0 if task == "train" else 1)
    assert summary["training"]["mode"] == "central"


def test_training_seed_matrix_uses_same_external_data_pool(author_files):
    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    experiment = training_experiment(documents)
    del experiment["factory"], experiment["workload"]
    experiment["matrix"] = {
        "factories": ["inputs/factory.yaml"],
        "workloads": ["inputs/workload.yaml"],
        "seeds": [101, 202],
    }
    write(paths["experiment"], experiment)
    plan = compile_experiment(paths["experiment"])
    assert len(plan.entries) == 2
    assert {entry.prepared.config.seed for entry in plan.entries} == {101, 202}
    assert (
        plan.entries[0].prepared.scenario_json == plan.entries[1].prepared.scenario_json
    )
    assert plan.entries[0].workload == plan.entries[1].workload
    assert (
        plan.entries[0].prepared.scientific_sha256
        != plan.entries[1].prepared.scientific_sha256
    )


def test_supported_learning_performance_check_does_not_calibrate_or_allocate(
    author_files, tmp_path
):
    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    write(paths["experiment"], training_experiment(documents, "train-evaluate"))
    for performance in ("recommend", "auto"):
        plan = compile_experiment(
            paths["experiment"], performance=performance, background=True
        )
        assert plan.experiment.execution.performance == performance
        assert plan.summary()["execution"]["executor"] == "tune"
        assert plan.experiment.execution.background
        assert len(plan.entries) == 1
        assert not (tmp_path / "results").exists()


def test_tune_with_calibration_disabled_is_rejected(author_files):
    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    experiment = training_experiment(documents, "train-evaluate")
    experiment["execution"] = {"executor": "tune", "performance": "off"}
    write(paths["experiment"], experiment)
    with pytest.raises(ConfigurationError, match="requires performance"):
        compile_experiment(paths["experiment"])


def test_input_hashes_capture_the_parsed_bytes_not_a_later_edit(
    author_files, monkeypatch
):
    import hashlib

    from smartsom.config import experiment_v4

    paths, _ = author_files
    original = paths["workload"].read_bytes()
    real = experiment_v4._read_source
    reads = []

    def read(path):
        result = real(path)
        reads.append(path)
        if path == paths["workload"]:
            path.write_text("schema: changed-after-read\n")
        return result

    monkeypatch.setattr(experiment_v4, "_read_source", read)
    plan = compile_experiment(paths["experiment"])
    frozen = json.loads(plan.entries[0].prepared.training_inputs_json)["authoring"]
    assert (
        frozen["input_sha256"][str(paths["workload"])]
        == hashlib.sha256(original).hexdigest()
    )
    assert reads.count(paths["workload"]) == 1


def test_author_progress_has_physical_training_and_validation_denominators(
    author_files,
):
    from smartsom.telemetry.workflow import describe_prepared

    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    experiment = training_experiment(documents, "train-evaluate")
    experiment["validation"] = {"enabled": True, "every_updates": 1, "replications": 1}
    write(paths["experiment"], experiment)
    prepared = compile_experiment(paths["experiment"]).entries[0].prepared
    view = describe_prepared(prepared, "train-evaluate")
    assert view["training_total"] == experiment["training"]["total_ticks"]
    assert view["training_unit"] == "physical ticks"
    assert view["round_size"] == experiment["training"]["ticks_per_update"]
    assert view["validation_cases"] == 1
    assert view["validation_rounds"] == view["round_total"]


def test_source_evaluation_preserves_each_frozen_replication(
    author_files, monkeypatch, tmp_path
):
    from dataclasses import asdict

    from smartsom.experiments import composable

    paths, documents = author_files
    documents["experiment"]["evaluation"]["replications"] = 3
    write(paths["experiment"], documents["experiment"])
    prepared = compile_experiment(paths["experiment"]).entries[0].prepared
    root = tmp_path / "saved"
    (root / "config").mkdir(parents=True)
    (root / "config/prepared.json").write_text(json.dumps(asdict(prepared)))
    monkeypatch.setattr(composable, "checkpoint_path", lambda *args: root)
    monkeypatch.setattr(composable, "evaluation_recipe", lambda p, checkpoint: p)
    options = prepared.config.evaluation.model_copy(
        update={"replications": 2, "deterministic": False}
    )
    result, _, _ = composable.prepare_evaluation(source=root, options=options)
    assert (
        json.loads(result.evaluation_json) == json.loads(prepared.evaluation_json)[:2]
    )
    assert not result.config.evaluation.deterministic
    for update in (
        {"replications": 4},
        {"seed": 7},
        {"scenarios": (str(paths["factory"]),)},
    ):
        with pytest.raises(ValueError, match="frozen data"):
            composable.prepare_evaluation(
                source=root,
                options=prepared.config.evaluation.model_copy(update=update),
            )


@pytest.mark.parametrize("status", ["completed", "recommended", "failed"])
def test_tune_saved_terminal_ledger_overrides_stale_running_display(status):
    from smartsom.experiments.author_driver import _publish_tune_state
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    display = RuntimeDisplay(DisplayOptions(progress="off"), kind="tune", quiet=True)
    display.configure_tuning(
        {
            "stage": "running",
            "entries": [
                {"experiment_id": "entry-0001", "status": "running", "actual_cpus": 1}
            ],
        }
    )
    display.update(
        "entry-0001",
        {"status": "running", "physical_ticks": 8},
        total=16,
        unit="physical ticks",
    )
    _publish_tune_state(
        display,
        {
            "status": status,
            "entries": {
                "entry-0001": {
                    "status": "queued" if status == "recommended" else status,
                    "physical_ticks": 0 if status == "recommended" else 16,
                }
            },
        },
    )
    display.finish(status)
    snapshot = display.snapshot()
    assert snapshot["tuning"]["stage"] == snapshot["status"] == status
    assert snapshot["tuning"]["entries"][0]["status"] == status
    assert snapshot["tasks"][0]["status"] == status
    if status == "recommended":
        assert snapshot["tasks"][0]["values"]["physical_ticks"] == 0


def test_explicit_entity_rule_override_and_unknown_id_validation(author_files):
    paths, documents = author_files
    algorithm = copy.deepcopy(documents["algorithm"])
    algorithm["agents"]["machine"]["overrides"] = {
        "M1": {
            "group": "machine-special",
            "policy": {"kind": "rule", "name": "normal_first"},
        }
    }
    write(paths["algorithm"], algorithm)
    plan = compile_experiment(paths["experiment"])
    frozen = json.loads(plan.entries[0].prepared.composition_json)
    assert frozen["bindings"]["machine"]["overrides"] == {"M1": "machine-special"}
    policies = json.loads(plan.entries[0].prepared.policies_json)
    assert policies["machine-special"]["implementation"]["name"] == "normal_first"
    algorithm["agents"]["machine"]["overrides"]["missing"] = algorithm["agents"][
        "machine"
    ]["overrides"].pop("M1")
    write(paths["algorithm"], algorithm)
    with pytest.raises(ConfigurationError, match="unknown machine override"):
        compile_experiment(paths["experiment"])


def test_conflicting_shared_policy_group_rejected(author_files):
    paths, documents = author_files
    algorithm = copy.deepcopy(documents["algorithm"])
    algorithm["agents"]["machine"]["overrides"] = {
        "M1": {"group": "machine", "policy": {"kind": "rule", "name": "normal_first"}}
    }
    write(paths["algorithm"], algorithm)
    with pytest.raises(ConfigurationError, match="sharing group has conflicting"):
        compile_experiment(paths["experiment"])


def test_optional_known_reliability_object_can_be_added_by_override(author_files):
    paths, _ = author_files
    plan = compile_experiment(
        paths["experiment"],
        sets=(
            "factory.reliability={enabled: true, defaults: {uptime: {distribution: uniform, min_ticks: 4, max_ticks: 4}, repair: {min_ticks: 2, max_ticks: 2}}}",
        ),
    )
    assert plan.entries[0].prepared.scenario.outages
    assert "reliability" not in yaml.safe_load(paths["factory"].read_text())


@pytest.mark.parametrize("option", ["background", "concurrency", "title"])
def test_execution_and_display_choices_do_not_change_scientific_identity(
    author_files, option
):
    paths, _ = author_files
    original = compile_experiment(paths["experiment"])
    kwargs = {
        "background": {"background": True},
        "concurrency": {"sets": ("experiment.execution.max_concurrent=2",)},
        "title": {"display_overrides": {"title": "A readable title"}},
    }[option]
    changed = compile_experiment(paths["experiment"], **kwargs)
    assert (
        original.entries[0].prepared.scenario_json
        == changed.entries[0].prepared.scenario_json
    )
    assert (
        original.entries[0].prepared.scientific_sha256
        == changed.entries[0].prepared.scientific_sha256
    )


def test_optimizer_change_affects_science_but_not_external_data(author_files):
    paths, documents = author_files
    write(paths["algorithm"], central_algorithm())
    write(paths["experiment"], training_experiment(documents))
    original = compile_experiment(paths["experiment"]).entries[0]
    changed = compile_experiment(
        paths["experiment"], sets=("algorithm.learner.parameters.learning_rate=0.0001",)
    ).entries[0]
    assert original.prepared.scenario_json == changed.prepared.scenario_json
    assert original.prepared.scientific_sha256 != changed.prepared.scientific_sha256


def test_held_out_data_are_shared_between_rules_and_central_learning_methods(
    author_files,
):
    paths, documents = author_files
    write(
        paths["workload"],
        {
            "schema": "smartsom.workload/v2",
            "profile": {
                "jobs": 8,
                "route": ["operation_1", "operation_3"],
                "nominal_min": 2,
                "nominal_max": 20,
                "due_at": 200,
            },
        },
    )
    rules = compile_experiment(paths["experiment"])
    rule_cases = json.loads(rules.entries[0].prepared.evaluation_json)
    write(paths["algorithm"], central_algorithm())
    write(paths["experiment"], training_experiment(documents, "train-evaluate"))
    central = compile_experiment(paths["experiment"])
    learned_cases = json.loads(central.entries[0].prepared.evaluation_json)
    assert len(rule_cases) == len(learned_cases) == 1
    assert rule_cases[0]["seed"] == learned_cases[0]["seed"]
    assert (
        rule_cases[0]["scenario"]["demands"] == learned_cases[0]["scenario"]["demands"]
    )
    assert rule_cases[0]["recipe"]["workload"] == learned_cases[0]["recipe"]["workload"]
