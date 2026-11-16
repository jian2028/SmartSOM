"""The learning WAIT boundary must preserve the physical decision protocol."""

from dataclasses import replace

import pytest

from smartsom.algorithms.base_policy import build_policy
from smartsom.algorithms.observation import build_observation
from smartsom.engine.production import ProductionSimulator
from smartsom.learning.factory_episode import FactoryEpisode
from smartsom.traffic import build_mover
from tests.unit.test_base_reservations import options, sim_with


def choose(policy, request, trace):
    action = policy.choose(request)
    selected = getattr(policy, "selected_identity", None)
    row = next(
        c
        for c in request.candidates
        if c.identity == selected or (selected is None and c.action == action)
    )
    trace.append(
        (request.tick, request.group, request.owner, request.stage, row.identity)
    )
    return row


@pytest.mark.parametrize(
    "mode,limit", [("finite", 3), ("finite", 1000), ("dynamic", 3)]
)
def test_wait_frames_preserve_complete_physics_and_semantic_decision_order(mode, limit):
    scenario = replace(sim_with(n=2).scenario, mode=mode, tick_limit=limit)
    settings = options()
    baseline = ProductionSimulator(scenario, contract="base")
    baseline.protocol.configure(settings)
    baseline.protocol.observation_transform = build_observation(settings)
    mover = build_mover(settings, baseline.protocol.routes, 101)
    policy = build_policy(settings, 101)
    expected = []
    while not baseline.done:
        stream = baseline.protocol.decisions()
        try:
            request = next(stream)
            while True:
                request = stream.send(choose(policy, request, expected).action)
        except StopIteration:
            pass
        baseline.protocol.advance(
            mover.moves(baseline.protocol.traffic_view(), baseline.tick)
        )

    episode = FactoryEpisode(scenario, settings, action_seed=101)
    policy = build_policy(settings, 101)
    actual, total = [], 0.0
    acted, seen = set(), set()
    empty_boundaries = 0
    while not (episode.frame.terminated or episode.frame.truncated):
        frame = episode.frame
        total += frame.team_reward
        if frame.request is not None:
            key = episode.subject_key(frame.request)
            acted.add(key)
            seen.add(key)
            episode.act(choose(policy, frame.request, actual).identity)
        else:
            assert all(row.tick == episode.sim.tick for row in frame.waiting)
            assert {episode.subject_key(row) for row in frame.waiting} == acted
            empty_boundaries += not frame.waiting
            acted.clear()
            assert episode._generator is None
            episode.wait()
    total += episode.frame.team_reward
    assert actual == expected
    assert episode.sim.snapshot() == baseline.snapshot()
    assert episode.sim.shipment_times == baseline.shipment_times
    assert episode.sim.total_reward == baseline.total_reward
    assert all(row.tick == episode.sim.tick for row in episode.frame.waiting)
    assert episode.frame.waiting  # Real bootstrap contexts, also at truncation.
    assert {episode.subject_key(row) for row in episode.frame.waiting} == seen
    report = episode.metrics()
    assert total == pytest.approx(report["learning_return"])
    assert total == pytest.approx(-report["observed_cost"])
    assert episode.frame.truncated == (limit == 3)
    assert report["truncated"] == episode.frame.truncated
    if limit == 1000:
        assert empty_boundaries > 0
        assert total == pytest.approx(-report["joint_cost"])
    with pytest.raises(ValueError):
        episode.wait()
    episode.close()


def test_illegal_actions_do_not_advance_pending_generator_or_time():
    episode = FactoryEpisode(sim_with(n=1).scenario, options())
    before = episode.sim.snapshot()
    pending = episode.frame
    with pytest.raises(ValueError, match="legal semantic"):
        episode.act("not-an-action")
    with pytest.raises(ValueError, match="physical boundary"):
        episode.wait()
    assert episode.frame is pending
    assert episode.sim.snapshot() == before
    episode.close()
