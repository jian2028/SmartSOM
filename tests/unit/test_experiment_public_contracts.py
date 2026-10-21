"""Small public compiler/ledger contracts, without subprocess termination."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from test_registered_observation import pair

from smartsom.config.codec import primitive
from smartsom.config.experiment_streams import training_episode_seed
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.production import named_seed
from smartsom.experiments import author_batch

ROOT = Path(__file__).resolve().parents[2]


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))


def inputs(tmp_path, algorithm="ppo", *, rules=False):
    write(
        tmp_path / "factory.yaml",
        yaml.safe_load((ROOT / "configs/test/factories/factory_hand.yaml").read_text()),
    )
    write(
        tmp_path / "workload.yaml",
        dict(
            schema="smartsom.workload/v3",
            demands=[
                dict(
                    demand_id="one",
                    steps=[
                        dict(
                            operation_id="op",
                            operation_type="operation_1",
                            nominal_ticks=2,
                        )
                    ],
                    due_at=0,
                )
            ],
        ),
    )
    if rules:
        method = yaml.safe_load(
            (ROOT / "configs/test/algorithms/all_rules_v2.yaml").read_text()
        )
    else:
        observation, network = pair()
        policy = dict(
            kind="new_model",
            extensions=dict(
                observation=primitive(observation),
                network_implementation=primitive(network),
                network={
                    branch: {"hidden_sizes": [8]}
                    for branch in (
                        ("actor", "critic") if algorithm == "ppo" else ("q",)
                    )
                },
            ),
        )
        method = dict(
            schema="smartsom.algorithm/v2",
            mode="resource",
            learner=dict(
                backend="rllib",
                algorithm=algorithm,
                gamma=1,
                parameters={"batch_size": 2},
            ),
            agents={
                role: {"default": copy.deepcopy(policy)}
                for role in ("machine", "buffer", "dispatcher")
            },
        )
        method["agents"]["mover"] = dict(
            default=dict(kind="rule", name="shortest_path")
        )
    write(tmp_path / "algorithm.yaml", method)
    document = dict(
        schema="smartsom.experiment-config/v4",
        task="evaluate" if rules else "train-evaluate",
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
        seed=101,
        data_seed=707,
        runtime=dict(environment=dict(mode="finite", tick_limit=4)),
        execution=dict(calibration_level="online", scheduling="adaptive"),
        validation=dict(enabled=False),
        evaluation=dict(replications=1, full_replay=False, verbose=False, record=False),
        output=dict(root=str(tmp_path / "runs"), name="tiny", tags=["medium"]),
    )
    if not rules:
        document["training"] = dict(total_ticks=4, ticks_per_update=4, max_ticks=4)
    path = tmp_path / "experiments/tiny.yaml"
    write(path, document)
    return path, document, method


def compiled(path):
    return compile_experiment(path).entries[0].prepared


@pytest.mark.parametrize("rules", [False, True])
def test_queued_native_work_has_frozen_phase_plan_and_comparable_costs(tmp_path, rules):
    from smartsom.experiments import author_driver
    from smartsom.telemetry.runtime import RuntimeDisplay

    path, document, _ = inputs(tmp_path, rules=rules)
    if not rules:
        document["evaluation"]["initial_control"] = True
        write(path, document)
    view = RuntimeDisplay(kind="author-plan", quiet=True)
    entries = []
    for seed, status in [(101, "completed"), (102, "queued")]:
        document["seed"] = seed
        write(path, document)
        root = author_driver.allocate(
            compile_experiment(path), root=tmp_path / f"plan-{seed}"
        )
        entry = json.loads((root / "plan.json").read_text())["entries"][0]
        entry = {**entry, "id": f"seed-{seed}"}
        assert not (root / "entries" / entry["id"] / "logs/progress.json").exists()
        author_driver._progress(view, root, entry, status, final=status == "completed")
        entries.append(entry["id"])
        if status == "completed":
            view.update(
                entry["id"],
                {
                    "status": status,
                    "execution": {
                        "schema": "smartsom.execution-observation/v1",
                        "attempt": "finished-cost-source",
                        "sequence": 1,
                        "updated_at": 100,
                        "training_ticks": 4,
                        "execution_ticks": 12,
                        "phases": {
                            "training": {"rate": 2},
                            "validation": {"rate": 2},
                            "evaluation": {"rate": 2},
                            "initial": {"rate": 2},
                            "loading": {"mean_seconds": 3},
                            "saving": {"mean_seconds": 1},
                            "finalization": {"mean_seconds": 4},
                        },
                    },
                },
                final=True,
            )
    completed, queued = (view.tasks[key] for key in entries)
    plan = queued["values"]["workflow"]
    assert plan["eta_group"] == completed["values"]["workflow"]["eta_group"]
    assert plan["mode"] == ("evaluation" if rules else "train-evaluate")
    assert plan["evaluation_horizons"] == [4]
    assert plan["control_names"] == ([] if rules else ["initial"])
    result = view.execution_telemetry.summarize(view, now=10, wall_time=100)
    estimate = result["tasks"][entries[1]]
    assert estimate["phase_work"]["evaluation"] == 4
    assert estimate["phase_work"].get("training", 0) == (0 if rules else 4)
    assert estimate["seconds"] is not None and estimate["seconds"] > 0
    assert estimate["missing"] == []
    assert result["batch"]["seconds"] == estimate["seconds"]


def test_explicit_roots_and_training_path_freeze_actual_seed(tmp_path):
    from smartsom.experiments.composable import TrainingSession

    path, document, _ = inputs(tmp_path)
    legacy = compiled(path)
    assert "seed_contract" not in json.loads(legacy.training_inputs_json)
    document.update(
        world_seed=11,
        environment_seed=22,
        policy_seed=33,
        training_environment_seeds="policy",
    )
    write(path, document)
    first = compiled(path)
    recipe = json.loads(first.training_inputs_json)
    streams = recipe["seed_contract"]
    assert streams["roots"] == dict(world=11, environment=22, policy=33)
    assert first.config.seed == streams["training_environment_root"] == 33
    assert recipe["data_seed"] == named_seed(11, "train:0")
    assert first.scenario.seed == named_seed(named_seed(22, "train:0"), "environment")
    session = TrainingSession.__new__(TrainingSession)
    session.prepared, session.config, session.settings = (
        first,
        first.config,
        first.config.training,
    )
    session.episodes = [0]
    assert session.new_sim(0).scenario.seed == named_seed(33, "environment:0:0")
    evaluation = json.loads(first.evaluation_json)
    document["policy_seed"] = 44
    write(path, document)
    second = compiled(path)
    assert (
        json.loads(second.evaluation_json)[0]["scenario"] == evaluation[0]["scenario"]
    )
    assert json.loads(second.training_inputs_json)["workload"] == recipe["workload"]
    assert training_episode_seed(
        json.loads(second.training_inputs_json), 44, 0, 0
    ) == named_seed(44, "environment:0:0")
    document["training_environment_seeds"] = "environment"
    write(path, document)
    third = compiled(path)
    assert training_episode_seed(
        json.loads(third.training_inputs_json), 44, 1, 2
    ) == named_seed(named_seed(22, "train:0"), "environment:1:2")


def test_legacy_seed_paths_and_invalid_frozen_root():
    assert training_episode_seed({}, 101, 2, 3) == named_seed(101, "training:2:3")
    assert training_episode_seed({"data_seed": 303}, 101, 2, 3) == named_seed(
        303, "environment:2:3"
    )
    with pytest.raises(ValueError, match="unsupported"):
        training_episode_seed({"seed_contract": {"schema": "unknown"}}, 101, 0, 0)


def frozen_inputs(tmp_path, *, rules=False):
    path, document, method = inputs(tmp_path, rules=rules)
    world = primitive(compiled(path).scenario)
    references = {}
    for phase, seed in (
        ("training", 101),
        ("validation", 303001),
        ("evaluation", 404001),
    ):
        snapshot = copy.deepcopy(world)
        snapshot["seed"] = seed
        snapshot["demands"][0]["due_at"] = seed
        write(
            tmp_path / f"{phase}.yaml",
            dict(
                schema="smartsom.frozen-production-world/v1",
                scenario=snapshot,
                provenance={"dataset": phase},
            ),
        )
        references[phase] = dict(
            world=f"../{phase}.yaml", data_seed=seed + 7, environment_seed=seed
        )
        if phase != "training":
            references[phase].update(id=phase + "_001", policy_seed=seed + 200000)
    document.pop("workload")
    document.pop("data_seed")
    document["runtime"].pop("environment")
    document["evaluation"].pop("replications")
    document["frozen_inputs"] = dict(
        schema="smartsom.frozen-experiment-inputs/v1",
        evaluation=[references["evaluation"]],
    )
    if not rules:
        document["training_environment_seeds"] = "legacy_policy"
        document["validation"] = dict(enabled=True, every_updates=1, full_replay=False)
        document["frozen_inputs"].update(
            training=references["training"], validation=[references["validation"]]
        )
    write(path, document)
    return path, document, method


def test_frozen_worlds_direct_cases_and_legacy_episode_rule(tmp_path):
    from smartsom.experiments.composable import TrainingSession

    path, document, _ = frozen_inputs(tmp_path)
    prepared = compiled(path)
    recipe = json.loads(prepared.training_inputs_json)
    assert recipe["seed_contract"]["schema"] == "smartsom.experiment-seed-streams/v2"
    session = TrainingSession.__new__(TrainingSession)
    session.prepared, session.config, session.settings = (
        prepared,
        prepared.config,
        prepared.config.training,
    )
    session.episodes = [0, 2]
    for index in (0, 1):
        expected = copy.deepcopy(recipe["frozen_world"])
        expected["seed"] = named_seed(
            101, f"training:{index}:{session.episodes[index]}"
        )
        assert primitive(session.new_sim(index).scenario) == expected
    for name, env_seed, policy_seed in (
        ("validation", 303001, 503001),
        ("evaluation", 404001, 604001),
    ):
        case = json.loads(getattr(prepared, name + "_json"))[0]
        assert (
            case["case"],
            case["replication"],
            case["seed"],
            case["scenario"]["seed"],
            case["policy_root"],
        ) == (
            name + "_001",
            0,
            env_seed + 7,
            env_seed,
            policy_seed,
        )
        assert case["recipe"]["frozen_world"] == case["scenario"]
        assert case["recipe"]["frozen_input_contract"]["world_sha256"]
    original = prepared.scientific_sha256
    document["frozen_inputs"]["evaluation"][0]["policy_seed"] += 1
    write(path, document)
    assert compiled(path).scientific_sha256 != original
    document["frozen_inputs"]["evaluation"][0]["policy_seed"] -= 1
    write(path, document)
    snapshot = yaml.safe_load((tmp_path / "evaluation.yaml").read_text())
    snapshot["provenance"]["revision"] = 2
    write(tmp_path / "evaluation.yaml", snapshot)
    assert compiled(path).scientific_sha256 != original


@pytest.mark.parametrize(
    "change",
    [
        "workload",
        "matrix",
        "data_seed",
        "world_seed",
        "environment_seed",
        "physics",
        "replications",
        "phase_seed",
        "duplicate_case",
        "missing_rule",
        "mismatched_seed",
        "mismatched_factory",
    ],
)
def test_frozen_input_conflicts_fail_before_allocation(tmp_path, change):
    path, document, _ = frozen_inputs(tmp_path)
    if change in ("workload", "matrix", "data_seed", "world_seed", "environment_seed"):
        document[change] = {
            "workload": "../workload.yaml",
            "matrix": {
                "factories": ["../factory.yaml"],
                "workloads": ["../workload.yaml"],
            },
            "data_seed": 0,
            "world_seed": 7,
            "environment_seed": 7,
        }[change]
    elif change == "physics":
        document["runtime"]["environment"] = dict(tick_limit=4)
    elif change in ("replications", "phase_seed"):
        document["evaluation"][
            "replications" if change == "replications" else "seed"
        ] = 1
    elif change == "duplicate_case":
        document["frozen_inputs"]["evaluation"] *= 2
    elif change == "missing_rule":
        document.pop("training_environment_seeds")
    elif change == "mismatched_seed":
        document["frozen_inputs"]["evaluation"][0]["environment_seed"] += 1
    else:
        snapshot = yaml.safe_load((tmp_path / "evaluation.yaml").read_text())
        snapshot["scenario"]["factory"]["grid"]["width"] += 1
        write(tmp_path / "evaluation.yaml", snapshot)
    write(path, document)
    with pytest.raises(ValueError):
        compiled(path)
    assert not (tmp_path / "runs").exists()


def test_case_policy_roots_reseed_action_streams_without_reinitializing_weights(
    tmp_path,
):
    import random

    import torch

    from smartsom.experiments.composable import policies_for

    path, _, _ = frozen_inputs(tmp_path)
    prepared = compiled(path)
    base, _ = policies_for(prepared)
    seeded, _ = policies_for(prepared, policy_seed=503001)
    for group, policy in seeded.items():
        seed = named_seed(503001, "policy:" + group)
        if hasattr(policy, "network"):
            assert all(
                torch.equal(value, base[group].network.state_dict()[key])
                for key, value in policy.network.state_dict().items()
            )
            assert torch.equal(
                policy.generator.get_state(),
                torch.Generator().manual_seed(seed % (2**63 - 1)).get_state(),
            )
            assert policy.random.getstate() == random.Random(seed).getstate()
        else:
            assert policy.seed == seed
            assert policy.rng.getstate() == random.Random(seed).getstate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_seed": 0},
        {"data_seed": 99},
        {"evaluation_overrides": {"replications": 1}},
        {"evaluation_overrides": {"replications": 2}},
    ],
)
def test_frozen_compiler_rejects_independent_effective_overrides(tmp_path, overrides):
    path, _, _ = frozen_inputs(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        compile_experiment(path, **overrides)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--data-seed", "99"],
        ["--replications", "2"],
        ["--preview"],
    ],
)
def test_frozen_public_cli_rejects_incompatible_case_flags(tmp_path, flags, capsys):
    from smartsom.experiments.cli import main

    path, _, _ = frozen_inputs(tmp_path, rules=True)
    assert main(["check", str(path), *flags]) == 2
    assert "frozen" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_frozen_public_cli_accepts_legal_default_resource_override(tmp_path, capsys):
    from smartsom.experiments.cli import main

    path, _, _ = frozen_inputs(tmp_path)
    assert (
        main(
            [
                "check",
                str(path),
                "--threads",
                "2",
                "--progress",
                "off",
                "--log-format",
                "json",
            ]
        )
        == 0
    )
    assert "checked" in capsys.readouterr().out
    prepared = (
        compile_experiment(path, sets=["experiment.runtime.numerical_threads=2"])
        .entries[0]
        .prepared
    )
    assert prepared.config.runtime.numerical_threads == 2
    assert not (tmp_path / "runs").exists()


def test_mixed_fresh_and_model_groups_initialize_only_selected_roles(tmp_path):
    from smartsom.experiments.composable import package
    from smartsom.learning.production_inference import build_groups

    path, document, method = inputs(tmp_path)
    original = compiled(path)
    policies, _ = build_groups(original, training=False)
    for group in ("machine", "dispatcher"):
        target = tmp_path / "weights" / group
        package(target, policies[group], original, group, 0, partners={})
        method["agents"][group]["default"] = dict(
            kind="model", model=dict(source=str(target))
        )
    write(tmp_path / "algorithm.yaml", method)
    default = compiled(path)
    assert default.config.training.groups == ("buffer",)
    document["training"]["groups"] = ["machine", "buffer"]
    write(path, document)
    selected = compiled(path)
    assert selected.config.training.groups == ("machine", "buffer")
    declarations = json.loads(selected.policies_json)
    assert declarations["machine"]["implementation"]["kind"] == "model"
    assert declarations["dispatcher"]["implementation"]["kind"] == "model"
    assert selected.config.interface_contract == "smartsom.configurable-experiment/v1"
    for groups in ([], ["machine", "machine"], ["unknown"], ["mover"]):
        document["training"]["groups"] = groups
        write(path, document)
        with pytest.raises(ValueError, match="training groups"):
            compiled(path)


def test_same_role_source_keeps_equal_weights_and_independent_group_rng(tmp_path):
    import torch

    from smartsom.experiments.composable import package
    from smartsom.learning.production_inference import build_groups

    path, document, method = inputs(tmp_path)
    original = compiled(path)
    source_policies, _ = build_groups(original, training=False)
    target = tmp_path / "initial-machine"
    package(target, source_policies["machine"], original, "machine", 0, partners={})
    model = dict(kind="model", model=dict(source=str(target)))
    method["agents"]["machine"]["default"] = model
    entity = original.scenario.factory.machines[-1].machine_id
    method["agents"]["machine"]["overrides"] = {
        entity: dict(group="machine_peer", policy=copy.deepcopy(model))
    }
    document["training"]["groups"] = ["machine", "machine_peer", "buffer", "dispatcher"]
    write(tmp_path / "algorithm.yaml", method)
    write(path, document)
    policies, _ = build_groups(compiled(path), training=False)
    first, second = policies["machine"], policies["machine_peer"]
    for key, value in first.network.state_dict().items():
        torch.testing.assert_close(
            value, second.network.state_dict()[key], rtol=0, atol=0
        )
    assert not torch.equal(first.generator.get_state(), second.generator.get_state())
    assert first.random.getstate() != second.random.getstate()
    assert first.encoder.state_dict() == second.encoder.state_dict()


@pytest.mark.parametrize("retained", [False, True])
def test_initial_control_records_initial_and_tune_reuses_control_spec(
    tmp_path, retained
):
    from smartsom.experiments.tuning_batch import BatchInputs, allocate_batch

    path, document, _ = inputs(tmp_path)
    document["evaluation"]["initial_control"] = True
    if retained:
        document["checkpointing"] = dict(retention=dict(mode="latest_full_and_best"))
    write(path, document)
    prepared = compiled(path)
    assert prepared.config.training.record_initial
    batch = BatchInputs(
        (
            dict(
                experiment_id="one",
                prepared=__import__("dataclasses").asdict(prepared),
                control_spec={},
            ),
        ),
        output_root=str(tmp_path / "tune"),
    )
    _, plan, _ = allocate_batch(batch)
    assert plan["entries"][0]["control_spec"]["names"] == ["initial"]


def test_rule_and_learning_use_unchanged_task_reward_contract(tmp_path):
    from smartsom.config.experiment_v3 import ShipmentTaskReward
    from smartsom.experiments.composable import evaluate_cases
    from smartsom.experiments.task_reward import configured_task_reward, shipment_reward

    path, document, _ = inputs(tmp_path / "rules", rules=True)
    legacy = compiled(path)
    document["task_reward"] = {}
    write(path, document)
    baseline = compiled(path)
    assert baseline.config.training is None
    assert configured_task_reward(baseline.config) == ShipmentTaskReward()
    assert baseline.scenario == legacy.scenario
    rows = evaluate_cases(baseline, json.loads(baseline.evaluation_json))
    assert rows[0]["task_reward_contract"] == primitive(ShipmentTaskReward())
    assert rows[0]["return"] != rows[0]["raw_legacy_return"]
    assert shipment_reward(
        ShipmentTaskReward(), dict(shipments=1, passing_change=1.0, overdue_time=0)
    ) == pytest.approx(0.5 / 128 + 0.3)
    path, document, method = inputs(tmp_path / "learning")
    document["task_reward"] = {}
    write(path, document)
    prepared = compiled(path)
    assert prepared.config.training.reward.task == configured_task_reward(
        baseline.config
    )
    method["learner"]["gamma"] = 0.99
    write(path.parent.parent / "algorithm.yaml", method)
    with pytest.raises(ValueError, match="gamma=1"):
        compiled(path)


def directory_inputs(tmp_path):
    path, document, _ = inputs(tmp_path)
    document["batch"] = dict(stage=1, parallel_files=2)
    first = path.with_name("dqn_a.yaml")
    document["algorithm"] = "../dqn.yaml"
    method = yaml.safe_load((tmp_path / "algorithm.yaml").read_text())
    method["learner"]["algorithm"] = "dqn"
    for role in ("machine", "buffer", "dispatcher"):
        method["agents"][role]["default"]["extensions"]["network"] = dict(
            q={"hidden_sizes": [8]}
        )
    write(tmp_path / "dqn.yaml", method)
    write(first, document)
    second = path.with_name("dqn_b.yaml")
    document["seed"] = 102
    write(second, document)
    third = path.with_name("ppo_a.yaml")
    document["batch"] = dict(stage=2, parallel_files=2, requires=["dqn_a"])
    document["algorithm"] = "../algorithm.yaml"
    write(third, document)
    path.unlink()  # fixture owns this tiny placeholder only
    return first.parent


def test_ordered_ids_and_tags_compile_only_selected_dependencies(tmp_path):
    directory = directory_inputs(tmp_path)
    write(
        directory / "small_cuda.yaml",
        dict(
            schema="smartsom.experiment-config/v4",
            output=dict(tags=["small"]),
            factory="missing.yaml",
        ),
    )
    plan = author_batch.compile_directory(
        directory, ids=["dqn_b", "dqn_a", "ppo_a"], tags=["medium"]
    )
    assert [row["id"] for row in plan.files] == ["dqn_b", "dqn_a", "ppo_a"]
    assert [row["stage"] for row in plan.files] == [1, 1, 2]
    assert plan.selection["resolved_ids"] == ["dqn_b", "dqn_a", "ppo_a"]
    filtered = author_batch.compile_directory(directory, tags=["medium"])
    assert len(filtered.files) == 3
    assert not (tmp_path / "runs").exists()
    for kwargs, message in (
        (dict(ids=["unknown"]), "unknown"),
        (dict(ids=["dqn_a", "dqn_a"]), "unique"),
        (dict(ids=[]), "nonempty"),
        (dict(tags=["nohit"]), "matched no"),
        (dict(ids=["dqn_a"], tags=["small"]), "does not match"),
        (dict(ids=["ppo_a"]), "missing selected dependency"),
    ):
        with pytest.raises(ValueError, match=message):
            author_batch.compile_directory(directory, **kwargs)


def test_cli_check_exposes_resolved_selection_and_rejects_unknown(tmp_path, capsys):
    from smartsom.experiments.cli import main

    directory = directory_inputs(tmp_path)
    assert (
        main(
            [
                "check",
                str(directory),
                "--id",
                "dqn_b",
                "--id",
                "dqn_a",
                "--tag",
                "medium",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["selection"]["resolved_ids"] == ["dqn_b", "dqn_a"]
    assert main(["check", str(directory), "--id", "unknown"]) == 2
    assert "unknown" in capsys.readouterr().err
    assert main(["check", str(directory / "dqn_a.yaml"), "--tag", "medium"]) == 2
    assert "directory check" in capsys.readouterr().err


def test_initial_native_stage_is_separate_idempotent_and_verified(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from smartsom import api
    from smartsom.experiments import author_driver, composable
    from smartsom.experiments.control import CURRENT as CONTROL
    from smartsom.experiments.evidence import write_json

    path, document, _ = inputs(tmp_path)
    document["evaluation"]["initial_control"] = True
    write(path, document)
    plan = compile_experiment(path)
    root = author_driver.allocate(plan, root=tmp_path / "owned-plan")
    saved = json.loads((root / "plan.json").read_text())
    entry = saved["entries"][0]
    worker = root / "entries" / entry["id"]
    worker.mkdir(parents=True)
    calls = []

    def train(prepared):
        directory = worker / "runs/training"
        directory.mkdir(parents=True)
        write_json(
            directory / "run.json",
            dict(
                kind="training",
                status="completed",
                scientific_sha256=prepared.scientific_sha256,
            ),
        )
        calls.append("training")
        return SimpleNamespace(run_dir=directory, status="completed")

    def evaluate(**kwargs):
        control = kwargs.get("purpose") == "initial_control"
        calls.append(("initial" if control else "primary", kwargs["selection"]))
        directory = worker / ("runs/initial" if control else "runs/primary")
        directory.mkdir(parents=True)
        write_json(
            directory / "run.json",
            dict(kind="evaluation", status="completed", purpose=kwargs.get("purpose")),
        )
        return SimpleNamespace(run_dir=directory, status="completed")

    monkeypatch.setattr(api, "train_prepared", train)
    monkeypatch.setattr(composable, "evaluate", evaluate)
    monkeypatch.setattr(author_driver, "boundary", lambda *args: None)
    monkeypatch.setattr(author_driver, "bind", lambda *args: None)
    token = CONTROL.set(SimpleNamespace(inherit_driver=lambda root: None))
    try:
        # Only the in-process stage function is exercised; no process lifecycle.
        worker_function = author_driver._worker.__wrapped__.__wrapped__
        worker_function(root, entry, worker)
        assert calls == ["training", ("primary", "last"), ("initial", "update-000000")]
        ledger = json.loads((worker / "stages.json").read_text())
        assert ledger["status"] == "completed"
        assert set(ledger["stages"]) == {"training", "evaluation", "initial_control"}
        author_driver.load(root)
        worker_function(root, entry, worker)
        assert len(calls) == 3
        ledger["stages"].pop("initial_control")
        write_json(worker / "stages.json", ledger)
        with pytest.raises(ValueError, match="required stages"):
            author_driver.load(root)
    finally:
        CONTROL.reset(token)


def test_multiple_stage_allocations_validate_whole_plan_and_frozen_selection(
    tmp_path, monkeypatch
):
    import smartsom.experiments.tuning_batch as tuning

    plan = author_batch.compile_directory(
        directory_inputs(tmp_path), ids=["dqn_b", "dqn_a", "ppo_a"], tags=["medium"]
    )
    monkeypatch.setattr(tuning, "preflight", lambda inputs: None)
    root = author_batch.allocate(plan)
    _, saved, state = author_batch.load(root)
    assert set(state["tune_directories"]) == {"1", "2"}
    assert saved["selection"]["resolved_ids"] == ["dqn_b", "dqn_a", "ppo_a"]
    # Resume reads the frozen plan, independent of the original directory.
    for path in plan.directory.glob("*.yaml"):
        path.unlink()
    author_batch.load(root)
    state["tune_directories"]["2"] = state["tune_directories"]["1"]
    (root / "batch.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Tune identities"):
        author_batch.load(root)


def test_stage_scope_preserves_parent_digest_and_entry_order(tmp_path):
    plan = author_batch.compile_directory(
        directory_inputs(tmp_path), ids=["dqn_b", "dqn_a", "ppo_a"]
    )
    dqn = author_batch._tune_inputs(
        plan, output_root=tmp_path / "p1", parent_digest="whole-plan", stage=1
    )
    ppo = author_batch._tune_inputs(
        plan, output_root=tmp_path / "p2", parent_digest="whole-plan", stage=2
    )
    assert [row["experiment_id"] for row in dqn.entries] == [
        "dqn_b__entry-0001",
        "dqn_a__entry-0001",
    ]
    assert [row["experiment_id"] for row in ppo.entries] == ["ppo_a__entry-0001"]
    assert (
        dqn.provenance["parent_plan_sha256"]
        == ppo.provenance["parent_plan_sha256"]
        == "whole-plan"
    )


@pytest.mark.parametrize("result", ["completed", "failed", "stopped"])
def test_stage_barrier_smoke_resume_and_failure_retry_are_scoped(
    tmp_path, monkeypatch, result
):
    from contextlib import nullcontext

    import smartsom.experiments.batch as locks
    from smartsom.experiments.control import StopRequested

    files = [
        dict(
            id=name,
            stage=stage,
            task="train-evaluate",
            entry_ids=["one"],
            parallel_files=2,
            gate=None,
        )
        for name, stage in (("dqn", 1), ("ppo", 2))
    ]
    plan = dict(files=files)
    # Runtime admission reads the frozen authored ceiling for each Tune stage.
    for stage in (1, 2):
        directory = tmp_path / f"p{stage}"
        directory.mkdir()
        (directory / "plan.json").write_text(
            json.dumps(dict(entries=[dict(baseline_concurrency=2)]))
        )
    state = dict(
        status="prepared",
        stage="prepared",
        files={row["id"]: dict(status="queued") for row in files},
        tune_directory=None,
        tune_directories={"1": "p1", "2": "p2"},
        calibration_by_stage={"1": "online", "2": "online"},
        calibration_status="online",
    )
    events = []
    monkeypatch.setattr(author_batch, "load", lambda root: (tmp_path, plan, state))
    monkeypatch.setattr(author_batch, "bind", lambda *args: None)
    # This scheduling test replaces process ownership; admission is tested separately.
    from smartsom.experiments import queue_admission

    monkeypatch.setattr(queue_admission, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_admission, "smoke_slot", lambda *args: nullcontext())
    monkeypatch.setattr(author_batch, "requested", lambda root: False)
    monkeypatch.setattr(author_batch, "_save", lambda *args: None)
    monkeypatch.setattr(author_batch, "_publish", lambda *args: None)
    monkeypatch.setattr(locks, "exclusive_lock", lambda path: nullcontext())
    monkeypatch.setattr(
        author_batch,
        "_smoke_all",
        lambda *args, stage=None: events.append(("smoke", stage)),
    )

    def execute(root, parent, saved, **kwargs):
        events.append(("train", saved["tune_directory"]))
        if saved["tune_directory"] == "p1" and result == "stopped":
            raise StopRequested("owned stage checkpoint")
        return dict(
            entries={
                f"{name}__one": dict(status=result if name == "dqn" else "completed")
                for name in ("dqn", "ppo")
            }
        )

    monkeypatch.setattr(author_batch, "_run_tune", execute)
    outcome = author_batch.execute_saved.__wrapped__(tmp_path)
    assert events[:2] == [("smoke", 1), ("train", "p1")]
    if result == "completed":
        assert events[2:] == [("smoke", 2), ("train", "p2")]
        assert outcome["status"] == "completed"
    else:
        assert len(events) == 2 and state["files"]["ppo"]["status"] == "queued"
        assert outcome["status"] == result
        if result == "failed":
            before = list(events)
            assert author_batch.execute_saved.__wrapped__(tmp_path)["retry_required"]
            assert events == before
        # A stopped group resumes the same stage; completed earlier groups skip.
        state["status"] = "stopped"
        state["files"]["dqn"]["status"] = "completed"
        events.clear()
        assert author_batch.execute_saved.__wrapped__(tmp_path)["status"] == "completed"
        assert events == [("smoke", 2), ("train", "p2")]


def test_mixed_rule_gate_defers_single_learning_smoke_and_calibration_until_retry(
    tmp_path, monkeypatch
):
    from smartsom.experiments import preflight
    from smartsom.experiments import tuning_batch as tuning

    path, document, _ = inputs(tmp_path)
    document["batch"] = dict(stage=20, requires=["rules"])
    document["execution"]["calibration_level"] = "quick"
    learning = path.with_name("learning.yaml")
    write(learning, document)
    rule = copy.deepcopy(document)
    rule.update(task="evaluate", algorithm="../rules.yaml")
    rule.pop("training")
    rule["batch"] = dict(
        stage=10,
        gate=dict(min_cases=1, min_deliveries_each=1, require_first_pickup=False),
    )
    write(path.parent / "rules.yaml", rule)
    write(
        tmp_path / "rules.yaml",
        yaml.safe_load(
            (ROOT / "configs/test/algorithms/all_rules_v2.yaml").read_text()
        ),
    )
    path.unlink()
    monkeypatch.setattr(tuning, "preflight", lambda inputs: None)
    root = author_batch.allocate(
        author_batch.compile_directory(path.parent, ids=["rules", "learning"])
    )
    _, plan, state = author_batch.load(root)
    assert set(state["tune_directories"]) == {"20"}
    tune_path = state["tune_directories"]["20"]
    assert (
        json.loads((root / tune_path / "plan.json").read_text())["provenance"]["stage"]
        == 20
    )
    identity = state["plan_sha256"]
    events = []
    monkeypatch.setattr(author_batch, "bind", lambda *args: None)
    # This scheduling test replaces process ownership; admission is tested separately.
    from smartsom.experiments import queue_admission

    monkeypatch.setattr(queue_admission, "initialize", lambda *args, **kwargs: None)
    from contextlib import nullcontext

    monkeypatch.setattr(queue_admission, "smoke_slot", lambda *args: nullcontext())
    monkeypatch.setattr(author_batch, "requested", lambda root: False)
    monkeypatch.setattr(author_batch, "_publish", lambda *args: None)

    def smoke(prepared, **kwargs):
        events.append(("smoke", "learning" if prepared.config.training else "rules"))
        return dict(status="passed")

    def rules(root, plan, state, rows, **kwargs):
        events.append(("run", "rules"))
        for row in rows:
            state["files"][row["id"]]["status"] = "completed"
        return True

    def gate_cases(*args):
        events.append(("gate", "rules"))
        if events.count(("gate", "rules")) == 1:
            raise OSError("temporary rule evidence read failure")
        return [dict(engineering_failure=False, delivered=1)]

    def tune(root, plan, state, *, recommend_only, **kwargs):
        assert state["tune_directory"] == tune_path
        events.append(("calibrate" if recommend_only else "train", "learning"))
        return dict(entries={"learning__entry-0001": dict(status="completed")})

    monkeypatch.setattr(preflight, "smoke", smoke)
    monkeypatch.setattr(author_batch, "_run_children", rules)
    monkeypatch.setattr(author_batch, "_evaluation_rows", gate_cases)
    monkeypatch.setattr(author_batch, "_run_tune", tune)
    execute = author_batch.execute_saved.__wrapped__
    assert execute(root)["status"] == "failed"
    assert events == [("smoke", "rules"), ("run", "rules"), ("gate", "rules")]
    _, _, failed = author_batch.load(root)
    assert failed["files"]["learning"]["status"] == "queued"
    assert failed["calibration_by_stage"]["20"] == "queued"
    assert set(json.loads((root / "smoke.json").read_text())["entries"]) == {
        "rules__entry-0001"
    }
    assert execute(root)["retry_required"]
    assert len(events) == 3
    assert execute(root, retry_failed=True)["status"] == "completed"
    assert events[3:] == [
        ("gate", "rules"),
        ("smoke", "learning"),
        ("calibrate", "learning"),
        ("train", "learning"),
    ]
    _, _, complete = author_batch.load(root)
    assert complete["plan_sha256"] == identity
    assert complete["calibration_by_stage"]["20"] == "completed"
    before = list(events)
    assert execute(root)["status"] == "completed"
    assert events == before + [("gate", "rules")]
