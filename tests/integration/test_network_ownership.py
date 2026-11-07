"""Owner routing, independent updates and exact full-state continuation."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.domain.observation_data import vector
from tests.helpers.observations import numeric_candidate, numeric_request
from tests.unit.test_algorithm_components import author, recipe


def test_split_dispatcher_ownership_and_transition_subjects(tmp_path):
    from smartsom.learning.dqn.replay import transitions
    from smartsom.learning.ownership import NetworkOwnership
    from smartsom.learning.subjects import model_groups

    shared, _ = options(tmp_path, "dqn")
    # The ownership test supplies concrete modules directly; full-network config
    # validation is exercised separately by component tests.
    shared = shared.model_copy(
        update={
            "production": shared.production.model_copy(
                update={
                    "policies": {
                        name: policy.model_copy(
                            update={"split_stages": ("dispatcher",)}
                        )
                        for name, policy in shared.production.policies.items()
                    }
                }
            )
        }
    )
    ownership = NetworkOwnership(shared)
    templates = torch.nn.ModuleDict(
        {key: torch.nn.Linear(2, 1) for _, key, _ in model_groups(shared)}
    )
    models = ownership.expand(templates)
    pickup = replace(request("agv", 0, "dispatcher"), stage="pickup")
    delivery = replace(pickup, tick=1, stage="delivery")
    next_pickup = replace(pickup, tick=3)
    other = replace(pickup, owner="other", tick=2)
    assert ownership.key(pickup) != ownership.key(delivery)
    assert ownership.key(pickup) == ownership.key(other)
    p, d = models[ownership.key(pickup)], models[ownership.key(delivery)]
    before = deepcopy(d.state_dict())
    optimizer = torch.optim.SGD(p.parameters(), lr=0.1)
    p(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert all(torch.equal(value, d.state_dict()[key]) for key, value in before.items())
    rows, omitted = transitions(
        [{"request": r, "index": 0} for r in (pickup, delivery, other, next_pickup)],
        [1.0, 2.0, 4.0, 8.0],
        terminal=True,
        scale=1.0,
        split_stages=("dispatcher",),
    )
    first = next(r for r in rows["dispatcher"] if r["request"] is pickup)
    assert first["next"] is next_pickup
    assert first["physical_dt"] == 3 and first["reward"] == 7
    assert omitted == 0
    delivery_row = next(r for r in rows["dispatcher"] if r["request"] is delivery)
    assert delivery_row["next"] is None and delivery_row["reward"] == 14


def request(owner, tick=0, group="machine"):
    return numeric_request(
        tick,
        group,
        owner,
        "process",
        (
            numeric_candidate("a", "a", vector(0.2, 0.4)),
            numeric_candidate("b", "b", vector(0.7, 0.1)),
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
            inputs=replace(
                first.inputs,
                social=tuple(
                    SocialCandidate(
                        c.identity,
                        (SocialLink("peer", "resource", vector(), (vector(1),)),),
                    )
                    for c in first.candidates
                ),
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
