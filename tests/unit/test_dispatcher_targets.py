"""Targets-only Dispatcher decisions, including idle and forced-choice paths."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.algorithms.production_composition import BoundaryCoordinator
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.engine.production import ProductionSimulator

ROOT = Path(__file__).resolve().parents[2]


def scenario(matrix):
    name = "small_rules_auto" if matrix else "template1_warmup"
    prepared = api.prepare(
        api.load_config(ROOT / "configs/test/runs" / (name + ".yaml")),
        training=False,
    )
    return prepared.scenario if matrix else prepared.resolved.scenario


def driver(sim):
    names = dict(
        machine="normal_first",
        buffer="edd",
        dispatcher="nearest",
        mover="automatic_travel" if sim.protocol.matrix else "shortest_path",
    )
    return BoundaryCoordinator(
        sim,
        {r: RulePolicy(r, name) for r, name in names.items()},
        {r: {"default": r} for r in names},
    )


@pytest.mark.parametrize("matrix", [False, True])
def test_no_targets_skips_policy_and_advances_physics(matrix, monkeypatch):
    original = scenario(matrix)
    release = original.tick_limit + 1
    future = tuple(
        replace(d, release_at=release, reveal_at=release, due_at=release + 600)
        for d in original.demands
    )
    sim = ProductionSimulator(
        replace(original, demands=future, mode="finite"), contract="v3"
    )
    controller = driver(sim)
    monkeypatch.setattr(sim.protocol, "dispatch_candidates", lambda vehicle: ())

    def forbidden(request):
        raise AssertionError("idle Dispatcher must not invoke any policy")

    monkeypatch.setattr(controller.policies["dispatcher"], "choose", forbidden)
    row = controller.tick()
    assert sim.tick == 1 and not sim.released
    assert row["actions"]["dispatchers"] == ()
    assert not any(r["role"] == "dispatcher" for r in row["decisions"])
    assert all(a["target"] is None for a in sim.agvs.values())
    from smartsom.learning.production_collection import PhysicalCollector

    collector = PhysicalCollector(("dispatcher",), 0.99)
    collector.collect(
        0,
        0,
        controller,
        row,
        controller.policies,
        {"dispatcher": row["reward"]},
        algorithm="ppo",
    )
    assert not collector.active and not collector.trajectories
    assert collector.counts["dispatcher"]["decisions"] == 0


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("omitted", [False, True])
def test_target_request_cannot_submit_none_or_be_omitted(matrix, omitted):
    sim = ProductionSimulator(scenario(matrix), contract="v3")
    requests = sim.protocol.begin()
    machines = {
        r.owner: r.candidates[0].action for r in requests if r.role == "machine"
    }
    targets = {
        r.owner: r.candidates[0].action for r in requests if r.role == "dispatcher"
    }
    assert targets
    vehicle = next(iter(targets))
    before = deepcopy(sim.agvs)
    if omitted:
        del targets[vehicle]
    else:
        targets[vehicle] = None
    with pytest.raises(ValueError, match="Dispatcher target"):
        sim.protocol.accept_proposals(machines, targets)
    assert sim.agvs == before
    sim.protocol.abort()


def test_single_target_model_is_legal_and_not_an_actor_choice():
    pytest.importorskip("torch")
    from smartsom.learning.production_inference import ModelPolicy
    from smartsom.learning.production_models import (
        CandidateNetwork,
        PublicEncoder,
        default_network,
    )

    sim = ProductionSimulator(scenario(True), contract="v3")
    requests = sim.protocol.begin()
    original = next(r for r in requests if r.role == "dispatcher")
    request = replace(original, candidates=original.candidates[:1])
    assert request.candidates[0].action is not None
    encoder = PublicEncoder(
        sim.factory, dict(time_scale=100, count_scale=100), role="dispatcher"
    )
    network = CandidateNetwork(
        encoder.context_size,
        default_network("ppo"),
        "rllib.resource_ppo",
        candidate_width=encoder.candidate_width,
    )
    policy = ModelPolicy(
        encoder, network, dict(role="dispatcher", algorithm="ppo"), 101, training=True
    )
    controller = driver(sim)
    controller.policies["dispatcher"] = policy
    assert controller.select(request) == request.candidates[0].action
    assert controller.records[-1]["actor_mask"] is False
    assert controller.records[-1]["log_probability"] == pytest.approx(0, abs=1e-6)
    # Critic-only observation remains possible while there is no action request.
    encoded, value = policy.value_input(
        "dispatcher", request.owner, request.observation
    )
    assert encoded["candidates"] == []
    assert __import__("math").isfinite(value)
    sim.protocol.abort()
