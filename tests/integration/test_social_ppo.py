"""Native actor updates, masks, stage selection and versioned continuation."""

import json
from dataclasses import replace

import pytest

from smartsom.config.base_diagnostic import BaseAlgorithm, BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
from smartsom.domain.social import SOCIAL_ARMS, SocialCandidate, SocialLink
from smartsom.experiments.base_diagnostic import episode

torch = pytest.importorskip("torch")
from smartsom.learning.base_ppo import (  # noqa: E402
    BasePPO,
    pack,
    recorded_request,
    request_from_dict,
)
from smartsom.learning.social_actor import Pool  # noqa: E402


def options(arm="all", pool="attention"):
    return BaseAlgorithm(
        schema="smartsom.base-algorithm/v1",
        social_arm=arm,
        social_pool=pool,
        epochs=1,
        width=8,
        empty_wait=True,
        reconstructed_inputs=True,
    )


@pytest.mark.parametrize(
    "arm,pool", [(arm, "attention") for arm in SOCIAL_ARMS] + [("all", "mean-max")]
)
def test_all_arms_real_episode_update_and_restore(arm, pool, tmp_path):
    opts = options(arm, pool)
    policy = BasePPO(opts, 83)
    initial = {k: v.clone() for k, v in policy.models.state_dict().items()}
    work = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=8,
        initial_orders=4,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    result, rewards, critic = episode(
        base_factory("small"), work, opts, policy, 123, "normal", 100, max_ticks=120
    )
    assert policy.records
    for r in policy.records:
        request = r["request"]
        assert (request.social is not None) == (
            (request.group, request.stage) in SOCIAL_ARMS[arm]
        )
        restored_request = request_from_dict(
            json.loads(json.dumps(recorded_request(request)))
        )
        assert json.dumps(recorded_request(restored_request)) == json.dumps(
            recorded_request(request)
        )
    sample = policy.records[0]
    continuation = [r["request"] for r in policy.records[:30]]
    losses = policy.update(
        [v / 1000 for v in rewards],
        terminal=result["status"] == "completed",
        final_critic=critic,
    )
    assert losses and policy.updates > 0
    changed = [
        k
        for k, v in policy.models.state_dict().items()
        if not torch.equal(v, initial[k])
    ]
    if arm != "base":
        assert any(".social." in k for k in changed)
    policy.save(tmp_path / "last.pt")
    restored = BasePPO(opts, 9)
    restored.load(tmp_path / "last.pt")
    assert all(
        torch.equal(v, restored.models.state_dict()[k])
        for k, v in policy.models.state_dict().items()
    )
    assert restored.choose(sample["request"], sample["critic"]) == policy.choose(
        sample["request"], sample["critic"]
    )
    if arm == "all":
        for request in continuation:
            assert restored.choose(request, critic) == policy.choose(request, critic)
        following_rewards = [0.1] * (max(r.tick for r in continuation) + 1)
        assert restored.update(
            following_rewards, terminal=False, final_critic=critic
        ) == policy.update(following_rewards, terminal=False, final_critic=critic)
        assert all(
            torch.equal(v, restored.models.state_dict()[k])
            for k, v in policy.models.state_dict().items()
        )
    if arm != "base":
        with pytest.raises(ValueError, match="contract"):
            BasePPO(options("base"), 9).load(tmp_path / "last.pt")


def dispatcher_request(stage="pickup", peers=True, enabled=True):
    candidates = (
        BaseCandidate("source0", ("s0",), vector(1, 2)),
        BaseCandidate("source1", ("s1",), vector(2, 3)),
    )
    if stage == "pickup":
        candidates += (BaseCandidate("EMPTY_WAIT", ("EMPTY_WAIT",), vector()),)
    social = (
        tuple(
            SocialCandidate(
                c.identity,
                ()
                if c.action == ("EMPTY_WAIT",)
                else (
                    SocialLink(
                        c.identity,
                        "vehicle",
                        vector(),
                        (vector(1, 0, 0.4, 0), vector(0, 1, 0, 0.5)) if peers else (),
                    ),
                ),
            )
            for c in candidates
        )
        if enabled
        else None
    )
    return BaseDecision(
        0,
        "dispatcher",
        "car",
        stage,
        candidates,
        vector(),
        ((), (), (), ()),
        social=social,
    )


@pytest.mark.parametrize("mode", ("attention", "mean-max"))
def test_permutations_padding_empty_sets_wait_and_backward(mode):
    opts = options("dispatcher-pickup", mode)
    policy = BasePPO(opts, 41)
    request = dispatcher_request()
    reversed_request = replace(
        request,
        candidates=tuple(reversed(request.candidates)),
        social=tuple(
            replace(
                c,
                links=tuple(
                    replace(link, peers=tuple(reversed(link.peers))) for link in c.links
                ),
            )
            for c in reversed(request.social)
        ),
    )
    model = policy.models["dispatcher"]
    empty = dispatcher_request(peers=False)
    disabled = dispatcher_request("delivery", enabled=False)
    batch = pack([request, reversed_request, empty, disabled], [vector()] * 4, 20)
    x, value = model(batch)
    assert torch.allclose(x[0], x[1].flip(0), atol=1e-6)
    single, _ = model(pack([request], [vector()], 20))
    assert torch.allclose(x[0], single[0], atol=1e-6)
    assert torch.isfinite(x[batch["mask"]]).all()
    assert not torch.allclose(x[0], x[2])
    assert not torch.isclose(x[0, -1], x[2, -1])  # WAIT reads peer changes.
    assert torch.equal(value, value[:1].expand_as(value))
    x[batch["mask"]].sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    pool = Pool(8, 8, mode)
    pooled = pool(
        torch.randn(2, 3, 8), torch.zeros(2, 3, dtype=torch.bool), torch.randn(2, 8)
    )
    assert torch.equal(pooled, torch.zeros_like(pooled))
    wait_only = replace(
        empty, candidates=(empty.candidates[-1],), social=(empty.social[-1],)
    )
    assert torch.isfinite(model(pack([wait_only], [vector()], 20))[0]).all()
    # A batch containing only a disabled stage still uses Base and needs no payload.
    base = BasePPO(options("base"), 41)
    b = pack([disabled], [vector()], 20)
    assert torch.equal(model(b)[0], base.models["dispatcher"](b)[0])


def test_common_base_weights_critic_and_old_checkpoint_compatibility(tmp_path):
    base = BasePPO(options("base"), 42)
    for arm in SOCIAL_ARMS:
        policy = BasePPO(options(arm), 42)
        assert all(
            torch.equal(v, policy.models.state_dict()[k])
            for k, v in base.models.state_dict().items()
        )
    base.save(tmp_path / "legacy.pt")
    payload = torch.load(tmp_path / "legacy.pt", weights_only=False)
    del payload["options"]["social_arm"], payload["options"]["social_pool"]
    torch.save(payload, tmp_path / "legacy.pt")
    restored = BasePPO(options("base"), 0)
    restored.load(tmp_path / "legacy.pt")
    assert all(
        torch.equal(v, restored.models.state_dict()[k])
        for k, v in base.models.state_dict().items()
    )
    with pytest.raises(ValueError, match="arm"):
        base.choose(dispatcher_request(), vector())
    with pytest.raises(ValueError, match="arm"):
        BasePPO(options("dispatcher"), 0).choose(
            dispatcher_request(enabled=False), vector()
        )


@pytest.mark.parametrize("mode", ("attention", "mean-max"))
@pytest.mark.parametrize(
    "group,stage",
    (
        ("machine", "process"),
        ("pre", "accept"),
        ("pre", "service"),
        ("post", "accept"),
        ("post", "service"),
        ("input", "accept"),
        ("input", "service"),
    ),
)
def test_job_relations_permutation_padding_and_stop(group, stage, mode):
    policy = BasePPO(options(group, mode), 43)
    candidates = (
        BaseCandidate("a", ("a",), vector(1)),
        BaseCandidate("b", ("b",), vector(2)),
    )
    stop = group in ("pre", "post") and stage == "accept"
    if stop:
        candidates += (BaseCandidate("STOP", ("STOP",), vector()),)
    links = (
        SocialLink(
            "job-a",
            "resource",
            vector(1),
            (vector(0.1, 0.2), vector(0.5, 0.4)),
            count=2,
        ),
        SocialLink("job-b", "station", vector(3), (vector(0.3, 0.8),), count=1),
    )
    request = BaseDecision(
        0,
        group,
        "owner",
        stage,
        candidates,
        vector(0.5),
        ((), (), (), ()),
        social=tuple(SocialCandidate(c.identity, links) for c in candidates),
    )
    permuted = replace(
        request,
        candidates=tuple(reversed(candidates)),
        social=tuple(
            replace(
                c,
                links=tuple(
                    replace(link, peers=tuple(reversed(link.peers)))
                    for link in reversed(c.links)
                ),
            )
            for c in reversed(request.social)
        ),
    )
    shorter = replace(request, candidates=candidates[:1], social=request.social[:1])
    model = policy.models[group]
    logits, _ = model(pack([request, permuted, shorter], [vector()] * 3, 20))
    assert torch.allclose(logits[0], logits[1].flip(0), atol=1e-6)
    singleton, _ = model(pack([shorter], [vector()], 20))
    assert torch.allclose(logits[2, :1], singleton[0], atol=1e-6)
    if stop:
        changed = replace(
            request,
            social=tuple(
                replace(
                    c,
                    links=tuple(
                        replace(link, peers=tuple(vector(10) for _ in link.peers))
                        for link in c.links
                    ),
                )
                for c in request.social
            ),
        )
        changed_logits, _ = model(pack([changed], [vector()], 20))
        assert not torch.isclose(logits[0, -1], changed_logits[0, -1])


def test_pre_stop_peer_storage_scales_with_real_relations():
    from smartsom.learning.social_actor import pack_social

    links = (
        SocialLink("downstream", "station", vector(), (vector(),) * 15, count=5),
        SocialLink("alternative", "resource", vector(), (vector(),) * 4, count=4),
    )
    candidates = tuple(
        BaseCandidate(str(i), (str(i),), vector()) for i in range(20)
    ) + (BaseCandidate("STOP", ("STOP",), vector()),)
    entries = tuple(SocialCandidate(c.identity, links) for c in candidates[:-1]) + (
        SocialCandidate("STOP", links * 20),
    )
    request = BaseDecision(
        0,
        "pre",
        "owner",
        "accept",
        candidates,
        vector(),
        ((), (), (), ()),
        social=entries,
    )
    packed = pack_social([request] * 2, 21, 20)
    # There are 80 real relations per decision, not 21*40 padded relations.
    assert packed["peers"].numel() <= 2 * 80 * 15 * 48
    assert sum(t.numel() * t.element_size() for t in packed.values()) < 2 * 1024**2
