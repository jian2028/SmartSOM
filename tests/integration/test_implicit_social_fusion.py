"""Implicit fusion contracts, real DQN updates, and independent expectations."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
from smartsom.experiments.reservation_runtime import episode
from smartsom.learning.models import make_models
from smartsom.representations.v2.concat import features, padded
from smartsom.representations.v2.network import Attention, Network, pack
from smartsom.representations.v2.observation import FEATURES, social_relations
from smartsom.representations.v2.rules import CATALOG
from smartsom.representations.v2.social_config import ConcatSlots, social_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("scale", ["small", "medium", "large"])
@pytest.mark.parametrize("ownership", ["shared", "independent"])
def test_concat_recipes_keep_worlds_learning_and_reward(scale, ownership):
    before = load_inputs(
        ROOT / f"experiments/v3/{scale}/isl-{ownership}/experiment.yaml"
    )
    after = load_inputs(
        ROOT / f"experiments/v3/{scale}/isl-concat-{ownership}/experiment.yaml"
    )
    a, b = before[3].model_dump(), after[3].model_dump()
    for key in ("observation", "network"):
        social = b["production"]["policies"]["dqn"][key]["params"]["social"]
        assert social["kind"] == "implicit" and social["fusion"] == "concat"
        slots = social["slots"]
        assert slots["input_jobs"] == before[2].orders
        assert slots["stations"] == {"small": 2, "medium": 5, "large": 10}[scale]
        b["production"]["policies"]["dqn"][key]["params"] = a["production"]["policies"][
            "dqn"
        ][key]["params"]
    assert a == b
    for attr in (
        "train_seeds",
        "train_episodes",
        "drain_ticks",
        "validation",
        "test",
        "sampling",
        "concurrency",
    ):
        assert getattr(before[0], attr) == getattr(after[0], attr)
    assert before[1] == after[1] and before[2] == after[2]


@pytest.mark.parametrize(
    "social",
    [
        {"kind": "explicit"},
        {"kind": "implicit"},
        {"kind": "implicit", "fusion": "concat"},
        {"kind": "none", "fusion": "attention"},
        {"kind": "implicit", "fusion": "attention", "slots": {}},
        {"kind": "implicit", "fusion": "unknown"},
    ],
)
def test_unsupported_social_contracts_fail(social):
    with pytest.raises(ValueError):
        social_config(social)


def test_legacy_and_new_configs_cannot_silently_conflict():
    with pytest.raises(ValueError, match="not both"):
        social_config({"kind": "none"}, False)
    with pytest.raises(ValueError):
        ConcatSlots(stations=True, vehicles=2, input_jobs=3, post_jobs=4)


@pytest.mark.parametrize("arm,enabled", [("base", False), ("isl", True)])
def test_named_interface_preserves_legacy_parameters(arm, enabled):
    _, _, _, options, _ = load_inputs(
        ROOT / f"experiments/v3/small/{arm}-shared/experiment.yaml"
    )
    document = options.model_dump(mode="json", by_alias=True)
    for key in ("observation", "network"):
        document["production"]["policies"]["dqn"][key]["params"] = {
            "social_enabled": enabled
        }
    legacy = BaseAlgorithm.model_validate(document)
    torch.manual_seed(101)
    named = make_models(options, value_head=False)
    torch.manual_seed(101)
    old = make_models(legacy, value_head=False)
    assert named.state_dict().keys() == old.state_dict().keys()
    assert all(
        torch.equal(v, old.state_dict()[k]) for k, v in named.state_dict().items()
    )


def test_concat_padding_masks_nested_jobs_and_rejects_overflow():
    values = torch.tensor([[[[1.0, 2.0], [999.0, 999.0]]]])
    mask = torch.tensor([[[True, False]]])
    assert padded(values, mask, 3, "peers").tolist() == [
        [[1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ]
    with pytest.raises(ValueError, match="exceeds configured"):
        padded(values, mask, 1, "peers")
    slots = ConcatSlots(stations=1, vehicles=1, input_jobs=2, post_jobs=2)
    data = {
        "jobs": torch.tensor([[[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]]]),
        "job_mask": torch.tensor([[[True, False]]]),
        "job_peers": torch.ones(1, 1, 2, 1, 9),
        "job_peer_mask": torch.ones(1, 1, 2, 1, dtype=torch.bool),
        "job_meta": torch.tensor([[[[0.05, 0.0], [99.0, 99.0]]]]),
    }
    actual = features(data, "input", "accept", slots)[0, 0]
    assert torch.allclose(
        actual[:16], torch.tensor([1.0, 2.0, 3.0, *([1.0] * 10), 0.05, 0.0, 1.0])
    )
    assert torch.equal(actual[16:], torch.zeros(16))


def test_concat_orders_vehicle_relations_by_semantic_identity():
    v = SimpleNamespace(f=object(), distance=10.0, time=20.0)
    rows = [("agv_z", 0, 30, 10), ("agv_self", 1, 0, 50), ("agv_a", 1, 0, 20)]
    v.vehicles = lambda owner: rows
    request = SimpleNamespace(group="dispatcher", owner="agv_self")
    candidate = SimpleNamespace(action=("post_a",))
    expected = {"peers": ((1.0, 0.0, 1.0), (0.0, 3.0, 0.5))}
    assert social_relations(v, request, candidate, ordered=True) == expected
    rows.reverse()
    assert social_relations(v, request, candidate, ordered=True) == expected


@pytest.mark.parametrize("ownership", ["shared", "independent"])
def test_concat_real_rollout_update_and_checkpoint(ownership, tmp_path):
    ex, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/v3/small/isl-concat-{ownership}/experiment.yaml"
    )
    document = options.model_dump(mode="json", by_alias=True)
    document["production"]["policies"]["dqn"]["learner"]["params"].update(
        batch_size=4,
        warmup_transitions=4,
        update_every=4,
        updates_per_transition=0.25,
    )
    owners = options._owner_ids
    options = BaseAlgorithm.model_validate(document)
    options._owner_ids = owners
    policy = build_policy(options, 101)
    assert not any(isinstance(m, Attention) for m in policy.learner.models.modules())
    before = deepcopy(policy.learner.models.state_dict())
    report, rewards, final = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        ex.drain_ticks,
        max_ticks=600,
        metrics_level="off",
    )
    assert report["decisions"] > 0
    records = list(policy.records)
    policy.update(rewards, terminal=False, final_critic=final)
    assert policy.updates > 0
    assert any(
        not torch.equal(v, policy.learner.models.state_dict()[k])
        for k, v in before.items()
    )
    path = tmp_path / "model.pt"
    policy.save(path)
    restored = build_policy(options, 999)
    restored.load(path)
    assert all(
        torch.equal(v, restored.learner.models.state_dict()[k])
        for k, v in policy.learner.models.state_dict().items()
    )
    # Real recorded requests retain their candidate masks through packing.
    assert records
    for record in records[:5]:
        request = record["request"]
        batch = pack([request], critics=None, options=options)
        assert batch["stages"][request.stage]["mask"].all()


def test_observation_network_configuration_mismatch_fails_before_rollout():
    _, _, _, options, _ = load_inputs(
        ROOT / "experiments/v3/small/isl-shared/experiment.yaml"
    )
    document = options.model_dump(mode="json", by_alias=True)
    document["production"]["policies"]["dqn"]["observation"]["params"] = {
        "social": {"kind": "none"}
    }
    with pytest.raises(ValueError, match="social configuration mismatch"):
        build_policy(BaseAlgorithm.model_validate(document), 101)


def test_concat_mixed_stages_match_singletons_and_ignore_nested_padding():
    # Independent review regression: two stages, unequal action/job counts.
    job = {
        "condition": (1.0, 2.0, 3.0),
        "downstream": {"peers": ((2.0,) * 9,), "terminal": 0.0},
    }

    def request(stage, relations):
        size = FEATURES["input." + stage] + len(CATALOG["input." + stage])
        return SimpleNamespace(
            group="input",
            stage=stage,
            candidates=[object()] * len(relations),
            inputs={
                "context": (0.1,) * 6,
                "candidates": [(0.2,) * size] * len(relations),
                "social": relations,
            },
        )

    requests = [
        request("accept", [{"jobs": (job, job)}, {"jobs": ()}]),
        request("service", [{"downstream": job["downstream"]}]),
        request("accept", [{"jobs": (job,)}]),
    ]
    options = SimpleNamespace(count_scale=20.0)
    model = Network(
        "input",
        social_enabled=True,
        value_head=False,
        concat_slots=ConcatSlots(stations=2, vehicles=2, input_jobs=3, post_jobs=3),
    )
    batch = pack(requests, critics=None, options=options)
    combined = model.actor(batch)
    assert batch["stages"]["accept"]["mask"].tolist() == [[True, True], [True, False]]
    assert batch["stages"]["service"]["mask"].tolist() == [[True, False]]
    for index, row in enumerate(requests):
        single = model.actor(pack([row], critics=None, options=options))
        assert torch.allclose(
            combined[index, : len(row.candidates)], single[0], atol=1e-7
        )
    invalid = ~batch["stages"]["accept"]["job_mask"]
    for key in ("jobs", "job_meta", "job_peers"):
        batch["stages"]["accept"][key][invalid] = float("nan")
    assert torch.allclose(combined, model.actor(batch), atol=1e-7)


def test_fusion_checkpoint_identities_are_not_interchangeable(tmp_path):
    from smartsom.learning.state import load, save

    options = {
        arm: load_inputs(ROOT / f"experiments/v3/small/{arm}-shared/experiment.yaml")[3]
        for arm in ("base", "isl", "isl-concat")
    }
    for origin, original in options.items():
        path = tmp_path / f"{origin}.pt"
        save(path, original, {"origin": origin})
        for destination, target in options.items():
            if origin == destination:
                assert load(path, target) == {"origin": origin}
            else:
                with pytest.raises(ValueError, match="contract mismatch"):
                    load(path, target)
