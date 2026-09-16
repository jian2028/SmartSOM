"""Real grid resource PPO, semantic batch order, and four-role update audit."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.config import resolve_training_run
from smartsom.experiments import train_one

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.marl


@pytest.fixture(scope="module")
def backend():
    missing = [
        p for p in ("ray", "torch", "gymnasium") if importlib.util.find_spec(p) is None
    ]
    if missing:
        if os.environ.get("SMARTSOM_REQUIRE_MARL") == "1":
            pytest.fail(f"required MARL extras are missing: {missing}")
        pytest.skip("optional MARL extras are not installed")


def test_protocol_constructor_spaces_and_global_lifecycle_do_not_skip_episode_zero(
    backend,
):
    from ray.rllib.utils.pre_checks.env import check_multiagent_environments

    from smartsom.config.training import episode_root
    from smartsom.learning.production_ray import ResourceProductionEnv, role_mapping

    prepared = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    recipe = prepared.resolved
    calls = []

    def episode(index):
        calls.append(index)
        return recipe.episode(episode_root(101, index))

    config = {
        "scenario": recipe.scenario,
        "max_jobs": recipe.algorithm.max_jobs,
        "gamma": recipe.algorithm.gamma,
        "episode_source": episode,
    }
    env = ResourceProductionEnv(config)
    assert env.adapter.episode_index == -1 and env.adapter.sim is None
    assert calls == []
    assert {role_mapping(a) for a in env.possible_agents} == {
        "machine_policy",
        "agv_policy",
        "buffer_policy",
        "quality_policy",
    }
    for aid in env.possible_agents:
        assert env.get_observation_space(aid) == env.observation_spaces[aid]
        assert env.get_action_space(aid) == env.action_spaces[aid]
    # The official checker samples without knowing the active owner's mask.
    for space in env.action_spaces.values():
        space.sample = lambda *args, **kwargs: next(
            i for i, valid in enumerate(env.adapter.action_masks()) if valid
        )
    check_multiagent_environments(env)
    assert calls == [0] and env.adapter.episode_index == 0
    assert env.adapter.scenario == recipe.episode(episode_root(101, 0))
    other = ResourceProductionEnv(config)
    observations, _ = other.reset(seed=999)
    assert calls == [0, 0]
    assert other.adapter.scenario == env.adapter.scenario
    assert set(observations) == {other._actor()}
    env.close()
    other.close()


def test_framework_batch_order_is_semantic_and_all_columns_stay_aligned(backend):
    from smartsom.learning.production_ray import SemanticBatchOrder

    episodes = [SimpleNamespace(id_="z"), SimpleNamespace(id_="a")]
    keys = [
        (e.id_, a, "machine_policy")
        for e in episodes
        for a in ("machine:M1", "machine:M2")
    ]
    batch = {
        "obs": {k: [i] for i, k in reversed(list(enumerate(keys)))},
        "actions": {k: [i + 10] for i, k in enumerate(keys)},
        "rewards": {k: [i + 20] for i, k in reversed(list(enumerate(keys)))},
    }
    ordered = SemanticBatchOrder()(batch=batch, episodes=episodes)
    assert all(list(items) == keys for items in ordered.values())
    assert [v[0] for v in ordered["obs"].values()] == list(range(4))
    assert [v[0] for v in ordered["actions"].values()] == list(range(10, 14))
    assert [v[0] for v in ordered["rewards"].values()] == list(range(20, 24))


def test_role_sampling_row_association_survives_hash_seed(backend):
    code = """
import json, torch
from types import SimpleNamespace
from smartsom.learning.production_ray import SemanticBatchOrder
keys = {('episode', f'agv:t-{i}', 'agv_policy') for i in range(4)}
batch = {'obs': {k: [k[1]] for k in keys}}
batch = SemanticBatchOrder()(batch=batch, episodes=[SimpleNamespace(id_='episode')])
torch.manual_seed(1892537964)
actions = torch.distributions.Categorical(probs=torch.tensor([[.2,.4,.4]]*4)).sample()
print(json.dumps([(k[1], int(a)) for k,a in zip(batch['obs'], actions)]))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=dict(os.environ, PYTHONHASHSEED=seed),
            text=True,
        )
        for seed in ("1", "42")
    ]
    assert outputs[0] == outputs[1]
    assert [a for a, _ in json.loads(outputs[0])] == [f"agv:t-{i}" for i in range(4)]


@pytest.fixture(scope="module")
def trained(backend, tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixed-resource-ppo")
    from smartsom.config.experiment import load_config, prepare

    config = load_config(ROOT / "configs/runs/learning_marl.yaml")
    config.output.root = str(directory / "training")
    config.validation.enabled = False
    config.logging.tensorboard = False
    config.logging.progress = "off"
    config.logging.verbose = False
    return directory, train_one(prepare(config))


def test_fixed_budget_reload_does_not_claim_unvisited_resource_updates(trained):
    from smartsom.experiments.training_audit import audit_training
    from smartsom.learning.production import LearnedProductionDriver

    sys.path.insert(0, str(ROOT / "scripts"))
    from validation.resource_acceptance import require_training_result

    _, result = trained
    audit = audit_training(result.run_dir)
    assert audit["environment_steps"] == 4096 and audit["agent_steps"] == 4096
    metadata = json.loads((result.checkpoint_dir / "checkpoint.json").read_text())
    changes = metadata["component_changes"]
    assert set(changes) == {
        "machine_policy",
        "agv_policy",
        "buffer_policy",
        "quality_policy",
    }
    assert any(values["actor"] and values["critic"] for values in changes.values())
    all_updated = all(
        values[part] for values in changes.values() for part in ("actor", "critic")
    )
    if all_updated:
        require_training_result(audit, result.run_dir)
    else:
        with pytest.raises(ValueError, match="all four resource actors and critics"):
            require_training_result(audit, result.run_dir)
    recipe = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml").resolved
    driver = LearnedProductionDriver(result.checkpoint_dir, recipe.scenario)
    assert set(driver.modules) == set(metadata["modules"])
    driver.env.close()


def test_ten_fixed_runs_preserve_partial_replay_and_strict_acceptance(trained):
    directory, result = trained
    output = directory / "evaluation"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_resource_learning.py"),
            "--development",
            "--training-dir",
            str(result.run_dir),
            "--output-dir",
            str(output),
            "--workers",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
    )
    report = json.loads((output / "report.json").read_text())
    assert len(report["evaluation"]) == 10
    assert (
        report["evidence_kind"] == "development"
        and report["accepted_source_sha"] is None
    )
    for replication in range(5):
        rows = [r for r in report["evaluation"] if r["replication"] == replication]
        assert {r["algorithm"] for r in rows} == {"builtin.spt", "rllib.resource_ppo"}
        assert len({r["world_sha256"] for r in rows}) == 1
    for row in report["evaluation"]:
        assert row["execution_replay"]["status"] in {"passed", "partial_verified"}
        if row["algorithm"] == "builtin.spt":
            assert row["status"] == "passed"
        if row["status"] != "passed":
            assert row.get("makespan") is None
    eligible = report["training_eligibility"]["status"] == "passed"
    qualified = all(row["status"] == "passed" for row in report["evaluation"])
    assert report["status"] == ("passed" if eligible and qualified else "failed")
    assert process.returncode == (0 if eligible and qualified else 1)
