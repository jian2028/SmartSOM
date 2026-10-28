"""Current control contracts and a frozen native-v2 physical reference."""

import hashlib
import io
import json
from dataclasses import replace

import pytest

from smartsom.algorithms.base_policy import build_policy
from smartsom.algorithms.base_rules import BaseRule
from smartsom.algorithms.central_rules import CentralRule
from smartsom.config.base_diagnostic import (
    BaseAlgorithm,
    BaseWorkload,
    generate_scenario,
)
from smartsom.config.base_layout import base_factory
from smartsom.domain.production import Outage
from smartsom.engine.central_control import ADVANCE
from smartsom.engine.central_environment import CentralEnvironment
from smartsom.experiments.reservation_runtime import episode


def inputs(**options):
    return (
        base_factory("small"),
        BaseWorkload(
            schema="smartsom.base-workload/v1",
            orders=8,
            initial_orders=4,
            mean_interarrival=4,
            base_ticks=dict.fromkeys("ABCD", 2),
        ),
        BaseAlgorithm(schema="smartsom.base-algorithm/v1", method="rule", **options),
    )


def test_recording_keeps_decisions_rewards_and_random_sequence(tmp_path):
    from smartsom.trace.current import CurrentRecording
    from smartsom.trace.production import Playback

    factory, workload, options = inputs()
    policies = [BaseRule("random", 91), BaseRule("random", 91)]
    traces = [io.StringIO(), io.StringIO()]
    report, rewards, _ = episode(
        factory,
        workload,
        options,
        policies[0],
        123,
        "high",
        1000,
        trace=traces[0],
        metrics_level="off",
    )
    with CurrentRecording(tmp_path / "replay") as recording:
        recorded, actual, _ = episode(
            factory,
            workload,
            options,
            policies[1],
            123,
            "high",
            1000,
            trace=traces[1],
            recording=recording,
            metrics_level="off",
        )
    assert policies[0].rng.getstate() == policies[1].rng.getstate()
    assert traces[0].getvalue() == traces[1].getvalue()
    assert actual == pytest.approx(rewards)
    assert recorded["joint_cost"] == pytest.approx(report["joint_cost"])
    playback = Playback(tmp_path / "replay")
    assert playback.last_tick == report["tick"]
    assert playback.row(playback.last_tick)["state"]["completed"]


@pytest.mark.parametrize(
    ("condition", "ticks", "digest"),
    [
        (
            "normal",
            144,
            "86e9f5457f2b58c3a4c1a59b131c464c4d3df5d4541d244a81a233935bcfeaf6",
        ),
        (
            "high",
            160,
            "6be2e2e0656458475b49f1fbafbd60c57b27de82fa0a902b95bd0f6975df3da2",
        ),
    ],
)
def test_base_matches_frozen_main_native_v2(condition, ticks, digest):
    factory, workload, options = inputs()
    trace = io.StringIO()
    report, rewards, _ = episode(
        factory,
        workload,
        options,
        BaseRule("rule", 7),
        123,
        condition,
        1000,
        trace=trace,
        metrics_level="off",
    )
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    for row in rows:
        for decision in row["decisions"]:
            decision["decision"].pop("social", None)
            decision["decision"].pop("inputs", None)
    normalized = "".join(json.dumps(row) + "\n" for row in rows)
    assert hashlib.sha256(normalized.encode()).hexdigest() == digest
    assert report["tick"] == ticks and report["completed"] == 8
    assert sum(rewards) == pytest.approx(report["observed_return"])


@pytest.mark.parametrize("condition", ["normal", "high"])
def test_centralized_shares_rollout_and_completes(condition):
    factory, workload, options = inputs(control="centralized")
    policy = build_policy(options, 7)
    report, rewards, _ = episode(
        factory,
        workload,
        options,
        policy,
        123,
        condition,
        1000,
        metrics_level="off",
    )
    assert report["status"] == "completed" and report["completed"] == 8
    assert sum(rewards) == pytest.approx(-report["joint_cost"])


def environment():
    f, w, a = inputs(control="centralized")
    return CentralEnvironment(generate_scenario(f, w, 123, "normal", 1000), a)


def assignment(env):
    return next(c for c in env.info()["candidates"] if c.kind == "transport")


def test_assignment_is_zero_time_exclusive_and_reserves_capacity():
    env = environment()
    selected = assignment(env)
    before = env.core.protocol.quota(selected.destination)
    _, reward, done, cut, info = env.step(selected.identity)
    assert info["physical_dt"] == reward == env.core.tick == 0
    assert not done and not cut
    assert env.core.protocol.quota(selected.destination) == before - 1
    assert all(
        c.job != selected.job and c.vehicle != selected.vehicle
        for c in info["candidates"]
        if c.kind == "transport"
    )
    state = env.observation()
    with pytest.raises(ValueError, match="stale"):
        env.step(selected.identity)
    assert env.observation() == state


@pytest.mark.parametrize("loaded", [False, True])
def test_fault_cancels_target_without_rolling_back_inventory(loaded):
    env = environment()
    selected = assignment(env)
    env.step(selected.identity)
    core, protocol = env.core, env.core.protocol
    vehicle = core.agvs[selected.vehicle]
    source = vehicle["target"]["owner"]
    # Put the assigned vehicle at a real source port for the one-tick service.
    cell = protocol.routes.ports[source][0]
    occupant = next(
        (
            v
            for v, a in core.agvs.items()
            if v != selected.vehicle and tuple(a["cell"]) == cell
        ),
        None,
    )
    if occupant:
        core.agvs[occupant]["cell"] = vehicle["cell"]
    vehicle["cell"] = cell
    if loaded:
        env.step(ADVANCE.identity)
        assert vehicle["job"] == selected.job
    machine = protocol.station[selected.destination]
    core.scenario = replace(
        core.scenario, outages=(Outage(machine, core.tick + 1, core.tick + 30),)
    )
    env.step(ADVANCE.identity)
    assert selected.vehicle not in env.control.assignments
    assert vehicle["target"] is None
    if loaded:
        assert vehicle["job"] == selected.job
        assert core.jobs[selected.job]["location"] == selected.vehicle
        assert any(c.vehicle == selected.vehicle for c in env.info()["candidates"])
    else:
        assert vehicle["job"] is None
        assert core.jobs[selected.job]["location"] == source
    protocol.check_reservations()


def test_public_facts_are_immutable_and_exclude_hidden_state():
    env = environment()
    facts = env.observation()
    with pytest.raises(TypeError):
        facts["tick"] = 99
    with pytest.raises(TypeError):
        facts["jobs"][0]["public_risk"] = 99
    for job in env.core.jobs.values():
        job["defective"] = not job["defective"]
    env.core.scenario = replace(
        env.core.scenario, outages=(Outage(next(iter(env.core.machines)), 9999, 10000),)
    )
    assert env.observation() == facts


def test_random_controller_has_reproducible_choices():
    env = environment()
    a, b = CentralRule("random", 41), CentralRule("random", 41)
    assert [
        a.choose(env.observation(), env.info()["candidates"]) for _ in range(20)
    ] == [b.choose(env.observation(), env.info()["candidates"]) for _ in range(20)]
