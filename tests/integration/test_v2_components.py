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
@pytest.mark.parametrize("method", ["base", "isl", "morris", "justin", "random"])
def test_v2_official_recipes_load_multicapability_factory(scale, method):
    ex, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/v2/{scale}/"
        f"{method + '-shared' if method in ('base', 'isl') else method}-101/experiment.yaml"
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


@pytest.mark.parametrize(
    "scale,owners", [("small", 45), ("medium", 111), ("large", 221)]
)
@pytest.mark.parametrize("arm", ["base", "isl"])
def test_v2_independent_recipes_match_shared_except_ownership(scale, owners, arm):
    for seed in (101, 102, 103):
        paths = {
            mode: ROOT / f"experiments/v2/{scale}/{arm}-{mode}-{seed}/experiment.yaml"
            for mode in ("shared", "independent")
        }
        shared, independent = (yaml.safe_load(paths[m].read_text()) for m in paths)
        assert independent["train_seeds"] == [seed]
        assert independent["batch"]["stage"] == seed - 100
        assert shared.pop("algorithm") != independent.pop("algorithm")
        assert shared["tracking"]["group"] == independent["tracking"]["group"]
        assert shared == independent
    _, _, _, options, _ = load_inputs(paths["independent"])
    policy = build_policy(options, 101)
    assert len(policy.learner.models) == owners
    pointers = set()
    for model in policy.learner.models.values():
        current = {p.data_ptr() for p in model.parameters()}
        assert not pointers & current
        pointers |= current


@pytest.mark.parametrize("scale", ["small", "medium", "large"])
def test_v2_unified_campaign_inventory_and_common_panel_mover(scale):
    directory = ROOT / f"experiments/v2/{scale}"
    expected = (
        {
            f"{arm}-{sharing}-{seed}"
            for arm in ("base", "isl")
            for sharing in ("shared", "independent")
            for seed in (101, 102, 103)
        }
        | {f"morris-{seed}" for seed in (101, 102, 103)}
        | {"justin-101", "random-101"}
    )
    paths = list(directory.glob("*/experiment.yaml"))
    assert {p.parent.name for p in paths} == expected
    reference = yaml.safe_load(
        (directory / "base-shared-101/experiment.yaml").read_text()
    )
    movers = []
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        assert raw["test"] == reference["test"]
        assert (path.parent / raw["scenario"]).resolve() == (
            directory / "base-shared-101" / reference["scenario"]
        ).resolve()
        algorithm = yaml.safe_load((path.parent / raw["algorithm"]).read_text())
        movers.append(algorithm["traffic"]["mover"])
        if path.parent.name in {"random-101", "justin-101"}:
            assert raw["run"]["task"] == "evaluate"
            assert "train_episodes" not in raw
            assert "checkpoints" not in raw["evaluation"]
    assert all(
        m == {"target": "smartsom.traffic.rule_mover.create_mover"} for m in movers
    )


def test_v2_random_rollout_uses_base_actions_and_seeded_legal_choices():
    ex, factory, workload, options, _ = load_inputs(
        ROOT / "experiments/v2/small/random-101/experiment.yaml"
    )
    base = yaml.safe_load((ROOT / "configs/algorithms/v2-base-ddqn.yaml").read_text())
    raw = yaml.safe_load((ROOT / "configs/algorithms/v2-base-random.yaml").read_text())
    assert {k: v for k, v in raw.items() if k != "production"} == {
        k: v for k, v in base.items() if k != "production"
    }
    policy = build_policy(options, 101)
    assert policy.learner is None
    choose = policy.choose
    requests = []

    def record(request, critic=None):
        action = choose(request, critic)
        assert action in [c.action for c in request.candidates]
        requests.append(request)
        return action

    policy.choose = record
    report, _, _ = episode(
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
    machine = next(r for r in requests if r.group == "machine")
    assert len(machine.candidates) == 21  # seven Base rules times three modes
    assert {r.stage for r in requests if r.group == "dispatcher"} == {
        "pickup",
        "delivery",
    }
    left, right = (build_policy(options, 17) for _ in range(2))
    first = [left.choose(machine) for _ in range(64)]
    assert first == [right.choose(machine) for _ in range(64)]
    assert len(set(first)) > 1
    assert policy.updates == 0
