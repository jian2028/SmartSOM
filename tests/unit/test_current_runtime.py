"""Current control contracts and the source-backed rule-credit physical reference."""

import io
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.algorithms.base_policy import build_policy
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
from smartsom.policies.rule import BaseRule
from tests.helpers.algorithm import algorithm_document, algorithm_options
from tests.helpers.geometry import reservation_factory


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
        algorithm_options(method="rule", **options),
    )


def test_rule_calibration_uses_fixed_matching_baselines(tmp_path):

    import yaml

    from smartsom import api

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    experiment = yaml.safe_load(source.read_text())
    for name in ("scenario",) if "scenario" in experiment else ("factory", "workload"):
        experiment[name] = str(source.parent / experiment[name])
    provider = tmp_path / "forbidden.py"
    provider.write_text(
        "def create_policy(*, seed, control):\n    raise AssertionError('fixed baseline loaded custom provider')\ncreate_policy.CONTRACT = 'smartsom.policy/v1'\n"
    )
    observation = tmp_path / "observation.py"
    observation.write_text(
        'def encode(facts, decision):\n    return {"current_time": facts["tick"]}\n'
        "def create_observation(): return encode\n"
        'create_observation.OUTPUT_SCHEMA = "custom.central/v1"\n'
    )
    algorithm = algorithm_document(
        {
            "method": "rule",
            "control": "centralized",
            "bindings": {"central.assign": "custom"},
            "providers": {"custom": {"path": str(provider)}},
            "observation": {"path": str(observation)},
        }
    )
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    experiment["algorithm"] = "algorithm.yaml"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    settings = [
        "scenario.jobs.orders.total=8",
        "scenario.jobs.orders.initial=4",
        "scenario.jobs.arrivals.mean_interval_ticks=4",
        "scenario.jobs.processing.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        "experiment.run.seconds=60",
        "experiment.test_cases=4",
        "experiment.run.split=test",
        "experiment.run.action_seeds=1",
        "experiment.run.methods=[rule, random]",
    ]
    operations = {"tracking": {"mode": "disabled"}}
    api.run_experiment(
        path,
        task="smoke",
        operations=operations,
        sets=settings
        + [
            f'experiment.run.output="{tmp_path / "smoke"}"',
            "experiment.run.smoke_ticks=10",
        ],
    )
    smoke = json.loads((tmp_path / "smoke/summary.json").read_text())
    assert {r["method"] for r in smoke["results"]} == {"rule", "random"}
    api.run_experiment(
        path,
        task="calibrate",
        operations=operations,
        sets=settings
        + [
            f'experiment.run.output="{tmp_path / "calibrate"}"',
            "experiment.run.calibration_cases=1",
        ],
    )
    calibration = tmp_path / "calibrate/calibration.json"
    report = json.loads(calibration.read_text())
    assert report["status"] == "frozen"
    assert len(report["results"]) == 8
    assert {r["method"] for r in report["results"]} == {"rule", "random"}
    # The public runtime no longer offers standalone evaluation. Calibration
    # physics remain testable without restoring that retired execution path.
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments.reservation import allocate, arguments
    from smartsom.experiments.reservation_runtime import frozen_drain

    plan = compile_reservation(
        path,
        task="calibrate",
        sets=settings
        + [
            "algorithm.control=base",
            "algorithm.production.bindings={}",
            "algorithm.production={default: rule, policies: {rule: {kind: policy, observation: {target: smartsom.observations.rules.create_observation}, implementation: {target: smartsom.policies.rule.create_policy}}}}",
            f'experiment.run.output="{tmp_path / "wrong-control"}"',
            f'experiment.run.calibration="{calibration}"',
        ],
    )
    root = allocate(plan, announce=False)
    args, _ = arguments(root)
    with pytest.raises(ValueError, match="matching completed calibration"):
        frozen_drain(args, load_inputs(args.experiment)[-1])


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


@pytest.mark.parametrize("condition", ["normal", "high"])
def test_current_rule_completes_with_legal_actions_and_consistent_rewards(condition):
    factory, workload, options = inputs()
    # ADR0099 deliberately changes speed decisions. Historical normal-only trace
    # hashes are not contracts for the current Rule; physical invariants remain.
    factory = reservation_factory()
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
    processing = []
    for row in rows:
        for decision in row["decisions"]:
            raw = decision["decision"]
            assert decision["action"] in [c["action"] for c in raw["candidates"]]
            if raw["group"] == "machine":
                processing.append(decision["action"])
    assert processing and {action[1] for action in processing} >= {"slow", "fast"}
    assert report["termination_reason"] == "all_delivered"
    assert report["completed"] == workload.orders == 8
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


def uncached_transports(control):
    """Per-vehicle transport candidates, recomputing every query."""
    core, protocol = control.core, control.protocol
    reserved = {a.job for a in control.assignments.values()}
    result = []
    for vehicle, state in core.agvs.items():
        if vehicle in control.assignments or state["service"] or state["target"]:
            continue
        jobs = (
            ((state["job"], None),)
            if state["job"]
            else [
                (j, owner)
                for owner in protocol.sources
                if protocol.eligible_owner(owner)
                for j in protocol.ready(owner)
                if j not in reserved
            ]
        )
        for job, source in jobs:
            if source and protocol.routes.to_owner(state["cell"], source) == math.inf:
                continue
            for destination in protocol.destination_owners(job):
                if (
                    protocol.eligible_owner(destination)
                    and protocol.quota(destination) > 0
                    and protocol.routes.to_owner(state["cell"], destination) < math.inf
                ):
                    result.append((job, vehicle, destination))
    return result


def test_candidate_reuse_matches_uncached_queries_across_assignments():
    env = environment()
    # One slot per station, so one assignment closes its destination.
    for owner, pools in env.core.capacity.items():
        if owner.startswith("machine_"):
            pools.update(dict.fromkeys(pools, 0))
    rng = random.Random(7)
    closed = 0
    for _ in range(200):
        candidates = env.info()["candidates"]
        assert [
            (c.job, c.vehicle, c.destination)
            for c in candidates
            if c.kind == "transport"
        ] == uncached_transports(env.control)
        selected = rng.choice(candidates)
        _, _, done, cut, _ = env.step(selected.identity)
        if done or cut:
            break
        closed += selected.kind == "transport" and not any(
            c.destination == selected.destination
            for c in env.info()["candidates"]
            if c.kind == "transport"
        )
    # Destinations close between calls, so reuse must not outlive one call.
    assert closed > 0


def with_central(options, **central):
    data = options.model_dump(mode="json", by_alias=True)
    return BaseAlgorithm.model_validate({**data, "central": central})


def test_central_settings_require_centralized_control():
    with pytest.raises(ValueError, match="centralized control"):
        with_central(algorithm_options(method="rule"), wip_limit=2)
    central = algorithm_options(method="rule", control="centralized")
    for invalid in ({"wip_limit": 0}, {}):
        with pytest.raises(ValueError):
            with_central(central, **invalid)
    # Unset keeps every persisted identity that earlier runs were saved with.
    assert "central" not in central.science_identity()
    data = central.model_dump(mode="json", by_alias=True)
    assert "central" not in data
    assert BaseAlgorithm.model_validate({**data, "central": None}) == central
    assert with_central(central, wip_limit=2).science_identity()["central"] == {
        "wip_limit": 2,
    }


def released_wip(control):
    core = control.core
    roles, jobs = core.roles, core.jobs
    return sum(
        j["location"] != "queue"
        and roles.get(j["location"]) not in ("system_input", "system_output")
        for j in jobs.values()
    ) + sum(
        roles.get(jobs[a.job]["location"]) == "system_input"
        for a in control.assignments.values()
    )


def test_wip_limit_withholds_only_input_releases():
    f, w, a = inputs(control="centralized")
    limit = 2
    env = CentralEnvironment(
        generate_scenario(f, w, 123, "normal", 1000), with_central(a, wip_limit=limit)
    )
    core = env.core
    roles = core.roles

    def at_input(job):
        return roles.get(core.jobs[job]["location"]) == "system_input"

    rng = random.Random(3)
    withheld = released_work = 0
    for _ in range(400):
        candidates = env.info()["candidates"]
        wip = released_wip(env.control)
        assert wip <= limit
        assert env.control.releases_open() == (wip < limit)
        transports = [c for c in candidates if c.kind == "transport"]
        if wip >= limit:
            assert not any(at_input(c.job) for c in transports)
            withheld += 1
            released_work += any(not at_input(c.job) for c in transports)
        _, _, done, cut, _ = env.step(rng.choice(candidates).identity)
        if done or cut:
            break
    # The limit was binding, and already released jobs still had transports.
    assert withheld > 0 and released_work > 0


def test_wip_limit_ignores_arrivals_queued_for_bounded_input():
    f, _, a = inputs(control="centralized")
    w = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=40,
        initial_orders=2,
        mean_interarrival=1,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    env = CentralEnvironment(
        generate_scenario(f, w, 123, "normal", 3000), with_central(a, wip_limit=3)
    )
    core = env.core
    for owner, pools in core.capacity.items():
        if core.roles[owner] == "system_input":
            pools.update(dict.fromkeys(pools, 2))
    rng = random.Random(3)
    queued = 0
    for _ in range(6000):
        queued = max(queued, len(core.queue))
        assert env.control.releases_open() == (released_wip(env.control) < 3)
        candidates = env.info()["candidates"]
        work = [c for c in candidates if c.kind != "advance"]
        _, _, done, cut, _ = env.step(rng.choice(work or candidates).identity)
        if done or cut:
            break
    # Queued arrivals exceeded the limit, yet releases kept the factory moving.
    assert queued > 3 and done and len(core.shipped) == w.orders


def test_episode_runtime_applies_wip_limit(monkeypatch):
    from smartsom.engine import central_control

    factory, workload, options = inputs(control="centralized")
    options = with_central(options, wip_limit=2)
    seen, mixed, original = [], [], central_control.CentralControl.candidates

    def candidates(self):
        result = original(self)
        seen.append((self.wip_limit, released_wip(self)))
        mixed.append(ADVANCE in result and len(result) > 1)
        return result

    monkeypatch.setattr(central_control.CentralControl, "candidates", candidates)
    report, _, _ = episode(
        factory,
        workload,
        options,
        build_policy(options, 7),
        123,
        "normal",
        1000,
        metrics_level="off",
    )
    assert {limit for limit, _ in seen} == {2}
    assert max(wip for _, wip in seen) == 2 and report["completed"] == 8
    # The engine keeps ADVANCE available alongside work.
    assert any(mixed)


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


def test_interrupted_full_recording_preserves_attempt_and_restarts(tmp_path):
    import time

    factory, workload, options = inputs()
    directory = tmp_path / "recording"
    with pytest.raises(TimeoutError):
        episode(
            factory,
            workload,
            options,
            BaseRule(),
            123,
            "normal",
            100,
            replay_directory=directory,
            wall_deadline=time.monotonic() - 1,
        )
    original = directory / "attempt-000001/replay/run.json"
    preserved = original.read_bytes()
    assert json.loads(preserved)["status"] == "failed"
    report, _, _ = episode(
        factory,
        workload,
        options,
        BaseRule(),
        123,
        "normal",
        100,
        replay_directory=directory,
    )
    assert original.read_bytes() == preserved
    assert report["recording_directory"] == str(directory / "attempt-000002")
    assert (directory / "attempt-000002/replay/run.json").is_file()


def test_evaluation_processes_are_operational_and_strict():
    from pydantic import ValidationError

    from smartsom.config.launch import Concurrency, scientific_identity

    document = {"experiment": {"concurrency": {"evaluation_processes": 1}}}
    expected = scientific_identity(document)
    document["experiment"]["concurrency"]["evaluation_processes"] = 4
    assert scientific_identity(document) == expected
    for bad in (0, -1, True, 1.5, 65):
        with pytest.raises(ValidationError):
            Concurrency(evaluation_processes=bad)
