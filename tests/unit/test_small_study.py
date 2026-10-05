"""Frozen study routing and actual small-budget backend updates, not a pilot."""

import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json, primitive
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments.composable import (
    TrainingSession,
    allocate,
    evaluate_cases,
    evaluation_recipe,
    prepared_from_run,
)

ROOT = Path(__file__).resolve().parents[2]


def prepared_small(name, transport, tmp_path):
    config = api.load_config(ROOT / f"configs/test/runs/{name}.yaml")
    if not name.startswith("central"):
        config.composition = str(
            ROOT
            / f"configs/test/compositions/small_train_{config.training.algorithm}.yaml"
        )
        config.training.groups = ("machine", "buffer", "dispatcher")
    config.scenario = str(
        ROOT / f"configs/test/scenarios/small_matrix_{transport}.yaml"
    )
    # Auto travel plus optional inspection must reach closed samples for every
    # role; this 64-tick PPO fixture ends before Machine receives a sample.
    needs_longer_horizon = config.training.algorithm == "dqn" or (
        name == "train_all_ppo" and transport == "auto"
    )
    config.training.total_ticks = 256 if needs_longer_horizon else 64
    config.training.ticks_per_update = config.training.total_ticks // 2
    config.training.record_initial = True
    config.validation.enabled = False
    config.evaluation.replications = 1
    config.evaluation.record = False
    config.evaluation.full_replay = True
    config.output.root = str(tmp_path)
    prepared = prepare_v3(config)
    parameters = json.loads(prepared.parameters_json)
    if config.training.algorithm == "dqn":
        parameters.update(batch_size=2, warmup_ticks=0, target_update_ticks=8)
    else:
        parameters.update(batch_size=4, n_epochs=1)
    return replace(prepared, parameters_json=canonical_json(parameters))


@pytest.mark.parametrize(
    "name", ["train_all_ppo", "train_all_dqn", "central_rllib_ppo", "central_sb3_ppo"]
)
@pytest.mark.parametrize("transport", ["auto", "zero"])
def test_actual_matrix_learning_and_exact_restore(name, transport, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    if "sb3" in name:
        pytest.importorskip("sb3_contrib")
    prepared = prepared_small(name, transport, tmp_path)
    root, record, frozen = allocate(prepared, "training")
    session = TrainingSession(frozen, root, record)
    result = session.execute(stop_after_updates=1)
    assert result.status == "interrupted"
    assert (root / "checkpoints/update-000000/groups").exists() or (
        root / "checkpoints/update-000000/controllers"
    ).exists()
    with (result.last_checkpoint / "continuation.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    session.execute()
    assert all(v > 0 for v in session.optimizations.values())
    continuous_actions = copy.deepcopy(session.actions)
    continuous_weights = {
        g: {k: v.clone() for k, v in p.network.state_dict().items()}
        for g, p in session.policies.items()
        if hasattr(p, "network")
    }
    resumed = TrainingSession(prepared_from_run(root), root, record)
    resumed.restore(saved)
    resumed.execute()
    assert resumed.actions == continuous_actions
    assert resumed.optimizations == session.optimizations
    for group, weights in continuous_weights.items():
        assert all(
            torch.equal(v, resumed.policies[group].network.state_dict()[key])
            for key, v in weights.items()
        )
    assert resumed.target_clock == session.target_clock
    for group in session.replays:
        assert (
            resumed.replays[group].state_dict() == session.replays[group].state_dict()
        )
    infer = evaluation_recipe(frozen, result.last_checkpoint)
    case = json.loads(prepared.evaluation_json)[0]
    # Bound this engineering check; completion is not required for a 32-tick case.
    case["scenario"]["tick_limit"] = 32
    rows = evaluate_cases(infer, [case])
    assert not rows[0]["engineering_failure"] and rows[0]["truncated"]
    assert all(not action["actions"]["movers"] for action in session.actions)


def test_full_preparation_has_paired_inputs_and_independent_cases(tmp_path):
    result = api.prepare_study(
        ROOT / "configs/test/studies/small_hv.yaml", tmp_path / "study"
    )
    assert result["training_runs"] == 24 and result["training_ticks"] == 393216
    assert (
        result["validation_episodes"] == 1920 and result["model_test_episodes"] == 120
    )
    directory = tmp_path / "study"
    plan = json.loads((directory / "plan.json").read_text())
    pools = set()
    for label in plan["datasets"]:
        pool = json.loads((directory / "datasets" / f"{label}.json").read_text())
        assert pool["pool_sha256"] not in pools
        pools.add(pool["pool_sha256"])
        assert (
            pool["levels"]["low"]["V"]
            < pool["levels"]["mid"]["V"]
            < pool["levels"]["high"]["V"]
        )
    by_v = {}
    for entry in plan["entries"]:
        frozen = prepared_from_run(directory / "snapshots" / entry["id"])
        config = frozen.config
        assert config.training.groups == ("machine", "buffer", "dispatcher")
        assert config.training.total_ticks == 16384
        assert (
            len(json.loads(frozen.validation_json))
            == len(json.loads(frozen.evaluation_json))
            == 5
        )
        demands = primitive(frozen.scenario.demands)
        if entry["V_case"] in by_v:
            assert demands == by_v[entry["V_case"]]
        by_v[entry["V_case"]] = demands
        assert len(frozen.scenario.demands) == 64
        assert frozen.scenario.seed == 101
        assert len(frozen.scenario.transport_matrix.points) == 48
    assert len(pools) == 11
    # A hand-authored generated config can change; batch still owns its saved snapshot.
    config_path = directory / plan["entries"][0]["config"]
    config_path.write_text("changed by test\n")
    assert api.show_study(directory)["training_runs"] == 24
    snapshot = directory / plan["entries"][0]["snapshot"]
    data = json.loads(snapshot.read_text())
    data["scientific_sha256"] = "changed"
    snapshot.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="snapshot changed"):
        api.show_study(directory)


def test_matrix_mixed_ppo_dqn_checkpoint_packages(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    import yaml

    from smartsom.experiments.composable import export

    sources = {}
    for name in ("train_all_ppo", "train_all_dqn"):
        prepared = prepared_small(name, "zero", tmp_path / name)
        root, record, frozen = allocate(prepared, "training")
        session = TrainingSession(frozen, root, record)
        session.execute()
        sources[name] = root
    dispatcher = export(
        sources["train_all_ppo"], tmp_path / "dispatcher.zip", group="dispatcher"
    )
    composition = yaml.safe_load(
        (ROOT / "configs/test/compositions/small_train_ppo.yaml").read_text()
    )
    selectors = {
        "machine": {
            "source": str(sources["train_all_ppo"]),
            "checkpoint": "update-000001",
            "group": "machine",
        },
        "buffer": {
            "source": str(sources["train_all_dqn"]),
            "checkpoint": "last",
            "group": "buffer",
        },
        "dispatcher": {"source": str(dispatcher)},
    }
    for role, selector in selectors.items():
        path = tmp_path / (role + ".yaml")
        path.write_text(
            yaml.safe_dump(
                {
                    "schema": "smartsom.policy/v1",
                    "role": role,
                    "implementation": {"kind": "model", "model": selector},
                }
            )
        )
        composition["groups"][role]["policy"] = str(path)
    composition["groups"]["mover"]["policy"] = str(
        ROOT / "configs/test/policies/mover_automatic_travel.yaml"
    )
    composition["pickup_matching"] = str(
        ROOT / "configs/test/rules/pickup_global_optimal.yaml"
    )
    comp_path = tmp_path / "mixed.yaml"
    comp_path.write_text(yaml.safe_dump(composition))
    config = api.load_config(ROOT / "configs/test/runs/small_rules_zero.yaml")
    config.composition = str(comp_path)
    config.scenario_overrides["tick_limit"] = 32
    config.output.root = str(tmp_path / "mixed")
    result = api.evaluate(config=config)
    assert result.engineering_failures == 0 and result.results[0]["truncated"]
    frozen = prepared_from_run(result.run_dir)
    declarations = json.loads(frozen.policies_json)
    assert declarations["buffer"]["resolved_model"]["metadata"]["algorithm"] == "dqn"
    assert declarations["machine"]["resolved_model"]["metadata"]["source_update"] == 1
    assert (
        declarations["dispatcher"]["resolved_model"]["metadata"]["role"] == "dispatcher"
    )
