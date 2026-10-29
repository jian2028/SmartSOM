"""Real current PPO/DQN and SL updates with exact safe-boundary restore."""

import pytest
import torch

from smartsom.algorithms.base_policy import build_policy
from smartsom.experiments.reservation_runtime import episode
from tests.helpers.algorithm import algorithm_options
from tests.unit.test_current_runtime import inputs


@pytest.mark.parametrize("method", ["ppo", "dqn"])
@pytest.mark.parametrize("arm", ["base", "all"])
def test_four_learning_combinations_restore_the_next_update(tmp_path, method, arm):
    factory, workload, _ = inputs()
    options = algorithm_options(
        schema="smartsom.base-algorithm/v1",
        method=method,
        social_arm=arm,
        epochs=1,
        target_update_interval=2,
    )
    policy = build_policy(options, 7)
    original = {k: v.clone() for k, v in policy.learner.models.state_dict().items()}
    report, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        123,
        "normal",
        500,
        max_ticks=100,
        metrics_level="off",
    )
    if arm == "base":
        assert all(r["request"].social is None for r in policy.records)
    else:
        assert any(r["request"].social for r in policy.records)
    policy.update(
        rewards, terminal=report["status"] == "completed", final_critic=critic
    )
    assert policy.updates > 0
    assert any(
        not torch.equal(v, original[k])
        for k, v in policy.learner.models.state_dict().items()
    )
    checkpoint = tmp_path / "model.pt"
    policy.save(checkpoint)
    restored = build_policy(options, 999)
    restored.load(checkpoint)
    if method == "dqn":
        assert restored.learner.truncated_tails > 0
        assert sum(map(len, restored.learner.replay.values())) > 0
        assert all(
            torch.equal(v, restored.learner.targets.state_dict()[k])
            for k, v in policy.learner.targets.state_dict().items()
        )
        assert not any("critic" in k for k in policy.learner.models.state_dict())
    for candidate in (policy, restored):
        report, rewards, critic = episode(
            factory,
            workload,
            options,
            candidate,
            456,
            "normal",
            500,
            max_ticks=100,
            metrics_level="off",
        )
        candidate.update(rewards, terminal=False, final_critic=critic)
    assert policy.updates == restored.updates
    assert all(
        torch.equal(v, restored.learner.models.state_dict()[k])
        for k, v in policy.learner.models.state_dict().items()
    )


def test_dqn_never_bootstraps_into_padding_or_an_unobserved_tail():
    from dataclasses import replace

    from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
    from smartsom.learning.dqn.learner import BaseDQN

    options = algorithm_options(
        schema="smartsom.base-algorithm/v1", method="dqn", epochs=1
    )
    learner = BaseDQN(options, 1)
    request = BaseDecision(
        0,
        "machine",
        "m",
        "process",
        (BaseCandidate("a", ("j", "normal"), vector()),),
        vector(),
        ((), (), (), ()),
    )
    learner.choose(request)
    learner.choose(replace(request, tick=2))
    learner.update([-2, -3, -4], terminal=False)
    assert learner.truncated_tails == 1
    row = learner.replay["machine"][0]
    assert row["reward"] == -5 and row["physical_dt"] == 2
    assert row["next"].tick == 2
    assert all(torch.isfinite(p).all() for p in learner.models.parameters())
