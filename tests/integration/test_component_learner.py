"""A file-defined learner uses real shared sampling, updates and full restore."""

from pathlib import Path

import torch

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.reservation_runtime import episode
from tests.unit.test_algorithm_components import author, recipe

ROOT = Path(__file__).resolve().parents[2]


def test_custom_learner_factory_updates_and_restores(tmp_path):
    data = recipe("ppo")
    data["production"]["policies"]["ppo"]["learner"] = {
        "path": str(ROOT / "examples/current/learner.py"),
        "factory": "create_learner",
        "params": {"epochs": 1},
    }
    path = author(tmp_path, data)
    plan = compile_reservation(
        path,
        sets=[
            "workload.orders=8",
            "workload.initial_orders=4",
            "workload.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        ],
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    _, factory, workload, options, _ = load_inputs(frozen / "experiment.yaml")
    policy = build_policy(options, 8)
    report, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        9,
        "normal",
        500,
        max_ticks=100,
        metrics_level="off",
    )
    policy.update(rewards, terminal=False, final_critic=critic)
    assert policy.last_losses["custom"]["batch_ticks"] == report["tick"]
    assert policy.updates > 0
    file = tmp_path / "model.pt"
    policy.save(file)
    restored = build_policy(options, 99)
    restored.load(file)
    assert restored.updates == policy.updates
    assert all(
        torch.equal(v, restored.learner.models.state_dict()[k])
        for k, v in policy.learner.models.state_dict().items()
    )
    # Sampler transfer is explicit; no runtime dependency on a .models attribute.
    sampler = build_policy(options, 1, training=False)
    sampler.learner.load_sampling_state(policy.learner.sampling_state())
    sampler.seed_sampling(123)
    assert all(
        torch.equal(v, sampler.learner.models.state_dict()[k])
        for k, v in policy.learner.models.state_dict().items()
    )
