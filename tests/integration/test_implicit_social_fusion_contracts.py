"""Research fusion contracts migrated from 1e91c81; no native learner migration."""

from types import SimpleNamespace

import pytest
import torch

from smartsom.representations.v2.concat import features, padded
from smartsom.representations.v2.network import Network, pack
from smartsom.representations.v2.observation import FEATURES, social_relations
from smartsom.representations.v2.rules import CATALOG
from smartsom.representations.v2.social_config import ConcatSlots, social_config


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
