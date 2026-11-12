"""E1a flat candidate packing, the idle ADVANCE mask and the shared snapshot."""

from pathlib import Path

import pytest
import torch
import yaml

from smartsom.algorithms.base_policy import build_policy
from smartsom.algorithms.training import TrainingComponents
from smartsom.config.base_diagnostic import BaseAlgorithm, load_inputs
from smartsom.controllers.e1a import network as e1a
from smartsom.engine.central_control import ADVANCE
from smartsom.experiments.reservation_runtime import episode

ROOT = Path(__file__).resolve().parents[2]


def configuration(**network):
    data = yaml.safe_load(
        (ROOT / "configs/algorithms/e1a-current-ppo.yaml").read_text()
    )
    data["production"]["policies"]["ppo"]["network"]["params"] = network
    return BaseAlgorithm.model_validate(data)


def rollout(options, ticks=400):
    ex, f, w, _, _ = load_inputs(
        ROOT / "experiments/implicit-sl/small/base/experiment.yaml"
    )
    policy = build_policy(options, 101)
    episode(
        f,
        w,
        options,
        policy,
        19,
        "normal",
        ex.drain_ticks,
        max_ticks=ticks,
        metrics_level="off",
    )
    return policy.learner, list(policy.learner.records)


def padded_logits(model, requests):
    """The former padded pack and actor, kept as the reference."""
    count = max(len(r.candidates) for r in requests)
    width = len(requests[0].inputs["candidates"][0])
    candidates = torch.zeros(len(requests), count, width)
    mask = torch.zeros(len(requests), count, dtype=torch.bool)
    associations = torch.full((len(requests), count, 3), -1, dtype=torch.long)
    owners = requests[0].inputs["owners"]
    positions = {owner: index for index, owner in enumerate(owners)}
    for i, r in enumerate(requests):
        candidates[i, : len(r.candidates)] = torch.tensor(r.inputs["candidates"])
        mask[i, : len(r.candidates)] = True
        associations[i, : len(r.candidates)] = torch.tensor(
            [[positions.get(o, -1) for o in row] for row in r.inputs["associations"]]
        )
    context = torch.tensor([r.inputs["global"] for r in requests])
    first = model.scorer[0]
    boundary = context.shape[-1]
    end = boundary + width
    hidden = torch.nn.functional.linear(
        context, first.weight[:, :boundary], first.bias
    ).unsqueeze(1)
    hidden = hidden + torch.nn.functional.linear(
        candidates, first.weight[:, boundary:end]
    )
    for slot in range(3):
        identities = associations[:, :, slot]
        columns = first.weight[
            :, end + slot * len(owners) : end + (slot + 1) * len(owners)
        ].T
        hidden = hidden + columns[identities.clamp(min=0)] * (
            identities >= 0
        ).unsqueeze(-1)
    scores = model.scorer[1:](hidden).squeeze(-1)
    return scores.masked_fill(~mask, -torch.inf)


@pytest.fixture(scope="module")
def plain():
    return rollout(configuration())


@pytest.fixture(scope="module")
def idle():
    return rollout(configuration(advance_only_when_idle=True))


def batch_of(rows):
    # Widest rows first, then a spread, so padding really differs per row.
    batch = sorted(rows, key=lambda r: -len(r["request"].candidates))[:12]
    return batch + rows[:: max(1, len(rows) // 40)]


def test_flat_pack_matches_padded_minibatch(plain):
    learner, rows = plain
    widths = sorted({len(r["request"].candidates) for r in rows})
    assert widths[0] == 1 and widths[-1] > 8 and len(widths) > 4
    model = next(iter(learner.models.values()))
    batch = batch_of(rows)
    requests = [r["request"] for r in batch]
    selected = torch.tensor([r["index"] for r in batch])
    results = []
    for logits_of in (
        lambda: model.actor(e1a.pack(requests, critics=None, options=None)),
        lambda: padded_logits(model, requests),
    ):
        model.zero_grad()
        logits = logits_of()
        distribution = torch.distributions.Categorical(logits=logits)
        (
            distribution.log_prob(selected).sum() + distribution.entropy().sum()
        ).backward()
        results.append(
            (
                logits.detach(),
                [p.grad.clone() for p in model.parameters() if p.grad is not None],
            )
        )
    (flat, flat_grads), (padded, padded_grads) = results
    assert torch.equal(flat.isinf(), padded.isinf())
    finite = ~flat.isinf()
    assert torch.allclose(flat[finite], padded[finite], atol=1e-6, rtol=1e-5)
    assert len(flat_grads) == len(padded_grads) > 0
    for left, right in zip(flat_grads, padded_grads, strict=True):
        assert torch.allclose(left, right, atol=1e-5, rtol=1e-4)


def test_network_param_selects_the_mask():
    assert (
        e1a.create_network(
            group="central", options=None, value_head=True
        ).advance_only_when_idle
        is False
    )
    model = e1a.create_network(
        group="central", options=None, value_head=True, advance_only_when_idle=True
    )
    assert model.advance_only_when_idle is True
    with pytest.raises(ValueError, match="true or false"):
        e1a.create_network(
            group="central",
            options=None,
            value_head=True,
            advance_only_when_idle="false",
        )
    with pytest.raises(ValueError, match="policy-gradient"):
        e1a.create_network(
            group="central", options=None, value_head=False, advance_only_when_idle=True
        )
    learner, _ = rollout(configuration(advance_only_when_idle=True), ticks=5)
    assert all(m.advance_only_when_idle for m in learner.models.values())


def test_idle_mask_hides_advance_only_while_work_exists(idle):
    learner, rows = idle
    model = next(iter(learner.models.values()))
    batch = batch_of(rows)
    requests = [r["request"] for r in batch]
    inputs = e1a.pack(requests, critics=None, options=None)
    with torch.no_grad():
        masked = model.actor(inputs)
        model.advance_only_when_idle = False
        try:
            unmasked = model.actor(inputs)
        finally:
            model.advance_only_when_idle = True
    seen = {"work": 0, "idle": 0}
    for i, request in enumerate(requests):
        kinds = [c.kind for c in request.candidates]
        work = [j for j, kind in enumerate(kinds) if kind != "advance"]
        advance = [j for j, kind in enumerate(kinds) if kind == "advance"]
        assert len(advance) == 1
        if work:
            seen["work"] += 1
            assert masked[i, advance[0]] == -torch.inf
            # Same distribution as an engine that never offered ADVANCE.
            expected = torch.log_softmax(unmasked[i, work], 0)
            assert torch.allclose(torch.log_softmax(masked[i], 0)[work], expected)
        else:
            seen["idle"] += 1
            assert torch.isfinite(masked[i, advance[0]])
    assert seen["work"] > 0 and seen["idle"] > 0
    # Sampled rollouts never chose ADVANCE while a work task was legal.
    for r in rows:
        if any(c.kind != "advance" for c in r["request"].candidates):
            assert r["action"] != ADVANCE.identity
    assert any(r["action"] == ADVANCE.identity for r in rows)


def test_plain_network_keeps_advance_beside_work(plain):
    _, rows = plain
    assert any(
        r["action"] == ADVANCE.identity
        and any(c.kind != "advance" for c in r["request"].candidates)
        for r in rows
    )


def test_critic_reuses_the_pending_request_snapshot(monkeypatch):
    original, shared = TrainingComponents.value_input, []

    def checked(self, protocol, context, snapshot=None):
        value = original(self, protocol, context, snapshot)
        if snapshot is not None:
            pending = protocol.central.pending_snapshot
            assert pending is not None and pending[1] is snapshot
        # A fresh snapshot of the same state gives the same critic input.
        assert value == original(self, protocol, context, None)
        shared.append(snapshot is not None)
        return value

    monkeypatch.setattr(TrainingComponents, "value_input", checked)
    learner, rows = rollout(configuration(), ticks=200)
    # Every decision's critic shared its actor snapshot; only the end-of-episode
    # bootstrap (no pending request) built its own.
    assert sum(shared) == len(rows) > 0
    assert len(shared) - sum(shared) <= 1


def test_pending_snapshot_serves_only_its_own_request():
    from dataclasses import replace

    options = configuration()
    ex, f, w, _, _ = load_inputs(
        ROOT / "experiments/implicit-sl/small/base/experiment.yaml"
    )
    from smartsom.config.base_diagnostic import generate_scenario
    from smartsom.engine.central_environment import CentralEnvironment

    env = CentralEnvironment(
        generate_scenario(f, w, 19, "normal", ex.drain_ticks), options
    )
    policy = build_policy(options, 101)
    protocol = env.core.protocol
    protocol.observation_transform = policy.observation_transform
    components = TrainingComponents(options)
    components.subjects = {}
    seen = []
    original = components.value_input

    def spy(protocol, context, snapshot=None):
        seen.append(snapshot)
        return original(protocol, context, snapshot)

    components.value_input = spy
    decisions = env.control.decisions()
    request = next(decisions)
    pending = env.control.pending_snapshot
    assert pending[0] is request and pending[1] is not None
    components.observe_critic(protocol, request)
    # An equal but different request object never borrows the snapshot.
    components.observe_critic(protocol, replace(request))
    assert seen[0] is pending[1] and seen[1] is None
    # Sending the action, or closing the generator, ends the pending window.
    work = next(c for c in request.candidates if c.kind != "advance")
    following = decisions.send(work.identity)
    assert env.control.pending_snapshot[0] is following
    components.observe_critic(protocol, request)
    assert seen[-1] is None
    decisions.close()
    assert env.control.pending_snapshot is None
