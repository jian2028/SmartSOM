"""Owner routing, independent updates and exact full-state continuation."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
from tests.unit.test_algorithm_components import author, recipe


def request(owner, tick=0, group="machine"):
    return BaseDecision(
        tick,
        group,
        owner,
        "process",
        (
            BaseCandidate("a", "a", vector(0.2, 0.4)),
            BaseCandidate("b", "b", vector(0.7, 0.1)),
        ),
        vector(0.1, 0.3),
        ((), (), (), ()),
    )


def options(tmp_path, method, arm="base"):
    data = recipe(method)
    spec = data["production"]["policies"][method]
    spec["learner"]["params"] = {
        "epochs": 1,
        "batch_size": 16,
        "target_update_interval": 2,
    }
    if arm != "base":
        spec["observation"] = {
            "target": "smartsom.representations.social.observation.create_observation",
            "params": {"arm": "all"},
        }
        spec["encoder"] = {
            "target": "smartsom.representations.social.encoder.create_encoder"
        }
    path = author(tmp_path, data)
    _, _, _, shared, _ = load_inputs(path)
    spec["parameter_sharing"] = {"roles": {"machine": "independent"}}
    spec["role_params"] = {"dispatcher": {"learning_rate": 0.0001}}
    path = author(tmp_path, data)
    _, _, _, independent, _ = load_inputs(path)
    return shared, independent


@pytest.mark.parametrize("method", ["ppo", "dqn"])
@pytest.mark.parametrize("arm", ["base", "social"])
def test_copied_initialization_isolation_and_resume(tmp_path, method, arm):
    shared, independent = options(tmp_path, method, arm)
    baseline = build_policy(shared, 101).learner
    policy = build_policy(independent, 101)
    learner = policy.learner
    owners = independent._owner_ids["machine"]
    first, other = request(owners[0]), request(owners[1])
    if arm == "social":
        from smartsom.domain.social import SocialCandidate, SocialLink

        first = replace(
            first,
            social=tuple(
                SocialCandidate(
                    c.identity,
                    (SocialLink("peer", "resource", vector(), (vector(1),)),),
                )
                for c in first.candidates
            ),
        )
    key = learner.ownership.key(first)
    untouched = learner.ownership.key(other)
    for k in (key, untouched):
        assert all(
            torch.equal(v, learner.models[k].state_dict()[n])
            for n, v in baseline.models["machine"].state_dict().items()
        )
    assert (
        next(learner.models[key].parameters()).data_ptr()
        != next(learner.models[untouched].parameters()).data_ptr()
    )
    assert learner.optimizers["dispatcher"].param_groups[0]["lr"] == 0.0001
    before = deepcopy(learner.models.state_dict())
    # Actor-independent test input still includes the unchanged global critic channel.
    critic = vector()
    policy.choose(first, critic)
    policy.choose(replace(first, tick=1), critic)
    policy.update([-1.0, -2.0], terminal=True, final_critic=critic)
    after = learner.models.state_dict()
    assert any(
        not torch.equal(v, after[n])
        for n, v in before.items()
        if n.startswith(key + ".")
    )
    assert all(
        torch.equal(v, after[n])
        for n, v in before.items()
        if not n.startswith(key + ".")
    )
    if method == "dqn":
        assert learner.replay[key] and not learner.replay[untouched]
        assert learner.key_updates[key] == 1 and learner.key_updates[untouched] == 0
    checkpoint = tmp_path / "model.pt"
    policy.save(checkpoint)
    restored = build_policy(independent, 987)
    restored.load(checkpoint)
    assert restored.learner.ownership.contract == learner.ownership.contract
    for active in (policy, restored):
        active.choose(first, critic)
        active.choose(replace(first, tick=1), critic)
        active.update([-2.0, -3.0], terminal=True, final_critic=critic)
    assert all(
        torch.equal(v, restored.learner.models.state_dict()[n])
        for n, v in learner.models.state_dict().items()
    )
    sampler = build_policy(independent, 22, training=False).learner
    sampler.load_sampling_state(learner.sampling_state())
    assert all(
        torch.equal(v, sampler.models.state_dict()[n])
        for n, v in learner.models.state_dict().items()
    )
    with pytest.raises(ValueError, match="ownership"):
        baseline.load_sampling_state(learner.sampling_state())


def test_dqn_redundant_role_override_preserves_targets(tmp_path):
    data = recipe("dqn")
    spec = data["production"]["policies"]["dqn"]
    spec["learner"]["params"] = {"epochs": 1, "target_update_interval": 2}
    path = author(tmp_path, data)
    _, _, _, plain, _ = load_inputs(path)
    spec["role_params"] = {"dispatcher": {"learning_rate": plain.learning_rate}}
    path = author(tmp_path, data)
    _, _, _, redundant, _ = load_inputs(path)
    policies = [build_policy(options, 101) for options in (plain, redundant)]
    for _ in range(2):
        for policy in policies:
            for group in ("machine", "dispatcher", "pre", "post", "input"):
                for tick in (0, 1):
                    policy.choose(
                        replace(
                            request("unused", tick, group),
                            stage={
                                "machine": "process",
                                "dispatcher": "pickup",
                                "pre": "accept",
                                "post": "accept",
                                "input": "accept",
                            }[group],
                        ),
                        vector(),
                    )
            policy.update([-1.0, -2.0], terminal=True, final_critic=vector())
        for name in ("models", "targets"):
            left, right = [getattr(p.learner, name).state_dict() for p in policies]
            assert all(torch.equal(value, right[key]) for key, value in left.items())
