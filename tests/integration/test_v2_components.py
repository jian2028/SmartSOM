"""V2 real decisions, learning, restoration and relation invariants."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
from smartsom.experiments.reservation_runtime import episode
from smartsom.representations.v2.network import Attention

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("arm", ["base", "isl"])
def test_v2_real_rollout_update_and_restore(arm, tmp_path):
    options = BaseAlgorithm.model_validate(
        yaml.safe_load((ROOT / f"configs/algorithms/v2-{arm}-ddqn.yaml").read_text())
    )
    ex, factory, workload, _, _ = load_inputs(
        ROOT / "experiments/implicit-sl/small/base/experiment.yaml"
    )
    policy = build_policy(options, 101)
    before = deepcopy(policy.learner.models.state_dict())
    report, rewards, final = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        ex.drain_ticks,
        max_ticks=200,
        metrics_level="off",
    )
    assert report["decisions"] > 0
    records = list(policy.records)
    assert {
        r["request"].stage for r in records if r["request"].group == "dispatcher"
    } == {"pickup", "delivery"}
    machine = next(r["request"] for r in records if r["request"].group == "machine")
    assert len(machine.candidates) == 21
    assert len({c.identity for c in machine.candidates}) == 21
    assert len(machine.inputs["context"]) == 8
    assert len(machine.inputs["candidates"][0]) == 11
    policy.update(rewards, terminal=False, final_critic=final)
    assert policy.updates > 0
    assert any(
        not torch.equal(value, policy.learner.models.state_dict()[key])
        for key, value in before.items()
    )
    for replay in policy.learner.replay.values():
        for transition in replay:
            request, following = transition["request"], transition["next"]
            if following is not None:
                assert (
                    request.owner == following.owner
                    and request.group == following.group
                )
                if request.group == "dispatcher":
                    assert request.stage == following.stage
    path = tmp_path / "checkpoint.pt"
    policy.save(path)
    restored = build_policy(options, 999)
    restored.load(path)
    request = records[0]["request"]
    assert policy.choose(request) == restored.choose(request)
    assert all(
        torch.equal(value, restored.learner.models.state_dict()[key])
        for key, value in policy.learner.models.state_dict().items()
    )


def test_attention_permutation_empty_and_masked_peer_invariance():
    attention = Attention("independent-test", 64, 9)
    query = torch.randn(2, 3, 64)
    peers = torch.randn(2, 3, 4, 9)
    mask = torch.tensor([[[True, True, False, False]] * 3, [[False] * 4] * 3])
    expected = attention(query, peers, mask)
    order = torch.tensor([3, 0, 2, 1])
    assert torch.allclose(
        expected, attention(query, peers[:, :, order], mask[:, :, order]), atol=1e-6
    )
    peers[~mask] = 10000
    assert torch.allclose(expected, attention(query, peers, mask), atol=1e-6)
    assert torch.equal(expected[1], torch.zeros_like(expected[1]))
    assert torch.isfinite(expected).all()


@pytest.mark.parametrize("scale", ["small", "medium", "large"])
@pytest.mark.parametrize("method", ["base", "isl", "morris", "justin"])
def test_v2_official_recipes_load_multicapability_factory(scale, method):
    ex, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/v2/{scale}/{method}-101/experiment.yaml"
    )
    assert len(factory.operation_types) == 8
    assert all(len(m.operation_types) == 2 for m in factory.machines)
    assert ex.test.cases_per_condition == 20
    assert len(ex.test.cases) == 80
    assert workload.optional_repeat_probability == 0.5


def test_v2_derived_cache_does_not_retain_finished_request():
    import gc
    import weakref
    from types import SimpleNamespace

    from smartsom.representations.v2.facts import View

    facts = SimpleNamespace(scales=(1, 1, 1), capacity={"pre": {"pool": 4}})
    view = View(facts)
    assert view.capacity("pre") == 4
    reference = weakref.ref(view)
    del view
    gc.collect()
    assert reference() is None


def test_v2_freezes_pickup_while_delivery_learns():
    raw = yaml.safe_load((ROOT / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    provider = next(iter(raw["production"]["policies"].values()))
    provider["frozen_stages"] = ["dispatcher.pickup"]
    options = BaseAlgorithm.model_validate(raw)
    ex, factory, workload, _, _ = load_inputs(
        ROOT / "experiments/implicit-sl/small/base/experiment.yaml"
    )
    policy = build_policy(options, 101)
    before = deepcopy(policy.learner.models.state_dict())
    _, rewards, final = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        ex.drain_ticks,
        max_ticks=200,
        metrics_level="off",
    )
    policy.update(rewards, terminal=False, final_critic=final)
    after = policy.learner.models.state_dict()
    assert all(
        torch.equal(value, after[key])
        for key, value in before.items()
        if key.startswith("dispatcher__pickup.")
    )
    assert any(
        not torch.equal(value, after[key])
        for key, value in before.items()
        if key.startswith("dispatcher__delivery.")
    )
    assert policy.learner.subject_transition_counts
    assert (
        sum(c["omitted"] for c in policy.learner.subject_transition_counts.values())
        == policy.learner.truncated_tails
    )
