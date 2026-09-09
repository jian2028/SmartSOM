"""Required MARL CI actually updates both role networks and audits fixed evaluation."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
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
        p
        for p in ("ray", "torch", "gymnasium", "pettingzoo")
        if importlib.util.find_spec(p) is None
    ]
    if missing:
        if os.environ.get("SMARTSOM_REQUIRE_MARL") == "1":
            pytest.fail(f"required MARL extras are missing: {missing}")
        pytest.skip("optional MARL extras are not installed")


def test_protocol_constructor_spaces_and_global_lifecycle_do_not_skip_episode_zero(
    backend,
):
    from ray.rllib.utils.pre_checks.env import check_multiagent_environments

    from smartsom.learning.rllib_resource import RLlibResourceEnv, policy_mapping

    resolved = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    env = RLlibResourceEnv({"resolved": resolved})
    assert env.parallel.episode_index == -1 and env.parallel.simulator is None
    for aid in env.possible_agents:
        assert env.get_observation_space(aid) == env.parallel.observation_space(aid)
        assert env.get_action_space(aid) == env.parallel.action_space(aid)
        assert policy_mapping(aid) in ("machine_policy", "agv_policy")
        # The official checker samples without a mask. Use the universally legal
        # NOOP on this isolated check instance; actual step validation is unchanged.
        env.action_spaces[aid].sample = lambda *args, **kwargs: 0
    check_multiagent_environments(env)
    assert env.parallel.episode_index == 0
    assert env.parallel.input == resolved.episode(0).input
    assert env.parallel.finished and env.parallel.agents == []
    assert env.agents == env.possible_agents
    other = RLlibResourceEnv({"resolved": resolved})
    obs, _ = other.reset(seed=999)
    assert other.parallel.episode_index == 0
    assert other.parallel.input == env.parallel.input
    assert set(obs) == set(env.possible_agents)


def test_framework_batch_order_is_semantic_and_all_columns_stay_aligned(backend):
    from smartsom.learning.rllib_resource import SemanticBatchOrder

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
from smartsom.learning.rllib_resource import SemanticBatchOrder
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
    resolved = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    resolved = replace(
        resolved,
        run=resolved.run.model_copy(
            update={"output_root": str(directory / "training")}
        ),
    )
    return directory, train_one(resolved)


def test_actual_two_role_updates_and_restoration(trained):
    _, trained_run = trained
    report = json.loads((trained_run.checkpoint_dir / "checkpoint.json").read_text())
    assert report["environment_steps"] == 4096
    assert report["agent_steps"] == 4096 * 12
    assert report["learner_updates"] == 16
    assert {r["role"] for r in report["role_weights"]} == {
        "machine_policy",
        "agv_policy",
    }
    assert all(r["initial_sha256"] != r["final_sha256"] for r in report["role_weights"])
    metrics = [
        json.loads(line)
        for line in (trained_run.run_dir / "learner_metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(metrics) == 16
    for role in ("machine_policy", "agv_policy"):
        assert all(
            any(role in k and "loss" in k for k in row["metrics"]) for row in metrics
        )
        # Updated hashes alone do not prove that the critic's clipped objective
        # remains trainable. The fixed micro must not saturate its value loss.
        assert all(
            row["metrics"][f"{role}/vf_loss"]
            == pytest.approx(row["metrics"][f"{role}/vf_loss_unclipped"])
            for row in metrics
        )


def test_ten_fixed_paired_runs_with_joint_action_and_schedule_replay(trained):
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
    assert process.returncode == 0, process.stdout[-8000:] + process.stderr[-4000:]
    report = json.loads((output / "report.json").read_text())
    assert (
        report["status"] == "passed"
        and report["completed"] == 10
        and report["failed"] == 0
    )
    assert len(report["evaluation"]) == 10
    assert report["evidence_kind"] == "development"
    assert report["accepted_source_sha"] is None
    assert sum("joint_replay" in r for r in report["evaluation"]) == 5
