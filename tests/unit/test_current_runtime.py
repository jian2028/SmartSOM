"""Current control contracts and the source-backed rule-credit physical reference."""

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.algorithms.base_policy import build_policy
from smartsom.algorithms.central_rules import CentralRule
from smartsom.config.base_diagnostic import (
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


def test_centralized_calibration_and_evaluation_use_fixed_matching_baselines(tmp_path):

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
    api.run_experiment(
        path,
        task="evaluate",
        operations=operations,
        sets=settings
        + [
            f'experiment.run.output="{tmp_path / "evaluate"}"',
            f'experiment.run.calibration="{calibration}"',
        ],
    )
    cases = [
        json.loads(line)
        for line in (tmp_path / "evaluate/cases.jsonl").read_text().splitlines()
    ]
    assert len(cases) == 8
    assert {r["method"] for r in cases} == {"rule", "random"}
    assert all(r["status"] == "completed" for r in cases)
    # The same workload cannot silently reuse centralized proof for Base physics.
    with pytest.raises(ValueError, match="matching completed calibration"):
        api.run_experiment(
            path,
            task="evaluate",
            operations=operations,
            sets=settings
            + [
                "algorithm.control=base",
                "algorithm.production.bindings={}",
                "algorithm.production={default: rule, policies: {rule: {kind: policy, observation: {target: smartsom.observations.rules.create_observation}, implementation: {target: smartsom.policies.rule.create_policy}}}}",
                f'experiment.run.output="{tmp_path / "wrong-control"}"',
                f'experiment.run.calibration="{calibration}"',
            ],
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
            142,
            "f4b3c10d56536e70d7b636d656e257cc0b7e8d0eea97c12a6ea35847defb8caf",
        ),
        (
            "high",
            167,
            "ac4bc812b72eba505ba9406364a6f52c69d9f8aaacf1ddb2f521774a8c861970",
        ),
    ],
)
def test_base_matches_frozen_cloud_server_credit_protocol(condition, ticks, digest):
    factory, workload, options = inputs()
    # Semantic trace regenerated independently from the verified pre-change archive
    # (5efd4d0 plus archived development changes, 2026-11-06). Input serialization
    # changed; physical events, candidate IDs/actions and decisions must not change.
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
    for row in rows:
        for decision in row["decisions"]:
            raw = decision["decision"]
            decision["decision"] = {
                key: raw[key]
                for key in ("tick", "group", "owner", "stage", "rejected_owners")
            }
            decision["decision"]["candidates"] = [
                {key: candidate[key] for key in ("identity", "action")}
                for candidate in raw["candidates"]
            ]
    normalized = json.dumps(rows, sort_keys=True, separators=(",", ":"))
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


def test_native_parallel_evaluation_preserves_order_and_resume(tmp_path):
    from smartsom import api
    from smartsom.experiments.reservation import arguments
    from smartsom.experiments.reservation_runtime import evaluate

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    common = [
        "experiment.run.task=evaluate",
        "scenario.jobs.orders.total=4",
        "scenario.jobs.orders.initial=2",
        "scenario.jobs.arrivals.mean_interval_ticks=4",
        "scenario.jobs.processing.base_ticks={A: 2, B: 2, C: 2, D: 2}",
        "experiment.drain_ticks=100",
        "experiment.drain_frozen=true",
        "experiment.run.seconds=null",
        "experiment.test_cases=4",
        "experiment.run.split=test",
        "experiment.run.action_seeds=1",
        "experiment.run.methods=[rule, random]",
        "experiment.run.selection=argmax",
    ]
    from smartsom.config.reservation import compile_reservation

    source = compile_reservation(source, sets=common).materialize(
        tmp_path / "standalone"
    )
    panels = []
    for workers in (1, 4):
        out = tmp_path / f"workers-{workers}"
        api.run_experiment(
            source,
            task="evaluate",
            operations={
                "tracking": {"mode": "disabled"},
                "concurrency": {"evaluation_processes": workers},
            },
            sets=common + [f'experiment.run.output="{out}"'],
        )
        rows = [
            json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()
        ]
        for row in rows:
            row.pop("wall_seconds")
            row.get("research_metrics", {}).pop("observer_seconds", None)
        panels.append(rows)
        receipts = {
            p.name: p.read_bytes() for p in (out / "case-commits").glob("case-*.json")
        }
        args, _ = arguments(out)
        evaluate(args)
        assert receipts == {
            p.name: p.read_bytes() for p in (out / "case-commits").glob("case-*.json")
        }
    assert len(panels[0]) == 8
    assert panels[0] == panels[1]


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


def _crash_evaluation_worker(connection, cancelled, settings):
    import os

    connection.recv()
    os._exit(7)


def test_parallel_worker_crash_is_not_a_case_result(tmp_path, monkeypatch):
    import multiprocessing
    from types import SimpleNamespace

    from smartsom.experiments import parallel_evaluation

    before = {p.pid for p in multiprocessing.active_children()}
    monkeypatch.setattr(parallel_evaluation, "_worker", _crash_evaluation_worker)
    prepared = (None, None, None, SimpleNamespace(component_items=lambda: ()))
    with pytest.raises(RuntimeError, match="exited without a result"):
        list(
            parallel_evaluation.case_results(
                {"output": str(tmp_path)},
                [{"index": i} for i in range(4)],
                prepared,
                workers=4,
                on_progress=lambda *_: None,
            )
        )
    assert {p.pid for p in multiprocessing.active_children()} <= before


def test_parallel_panel_pause_keeps_receipts_and_joins_workers(tmp_path, monkeypatch):
    import multiprocessing

    from smartsom import api
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import control, reservation_runtime
    from smartsom.experiments.reservation import arguments

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    out = tmp_path / "run"
    source = compile_reservation(
        source,
        sets=[
            "experiment.run.task=evaluate",
            "experiment.run.seconds=null",
            "experiment.run.methods=[rule]",
            "experiment.run.split=test",
            "experiment.test_cases=8",
            "experiment.run.action_seeds=1",
            "experiment.drain_ticks=100",
            "experiment.drain_frozen=true",
            "scenario.jobs.orders.total=4",
            "scenario.jobs.orders.initial=2",
            f'experiment.run.output="{out}"',
        ],
    ).materialize(tmp_path / "inputs")
    original = reservation_runtime.publish_progress
    requested = []

    def publish(root, payload, **kwargs):
        original(root, payload, **kwargs)
        if payload.get("completed_cases") == 1 and not requested:
            requested.append(True)
            control.request_pause(out)

    monkeypatch.setattr(reservation_runtime, "publish_progress", publish)
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(control.PauseRequested):
        api.run_experiment(
            source,
            operations={
                "tracking": {"mode": "disabled"},
                "concurrency": {"evaluation_processes": 4},
            },
        )
    assert requested
    assert json.loads((out / "progress.json").read_text())["status"] == "paused"
    receipts = {
        p.name: p.read_bytes() for p in (out / "case-commits").glob("case-*.json")
    }
    assert len(receipts) == 1
    assert {p.pid for p in multiprocessing.active_children()} <= before
    control.clear_pause(out)
    monkeypatch.setattr(reservation_runtime, "publish_progress", original)
    args, _ = arguments(out)
    reservation_runtime.evaluate(args)
    assert len(list((out / "case-commits").glob("case-*.json"))) == 8
    assert all(
        (out / "case-commits" / name).read_bytes() == data
        for name, data in receipts.items()
    )
    assert {p.pid for p in multiprocessing.active_children()} <= before


def _evaluation_failure_worker(connection, cancelled, settings):
    connection.recv()
    connection.send(("error", settings["error_type"], "injected worker failure"))
    connection.close()


@pytest.mark.parametrize(
    "name", ["ValueError", "AssertionError", "ResourceGuardExceeded"]
)
def test_parallel_failures_preserve_explicit_recovery_category(
    tmp_path, monkeypatch, name
):
    import multiprocessing
    from types import SimpleNamespace

    from smartsom.experiments import parallel_evaluation
    from smartsom.experiments.resource_guard import ResourceGuardExceeded

    expected = {
        "ValueError": ValueError,
        "AssertionError": AssertionError,
        "ResourceGuardExceeded": ResourceGuardExceeded,
    }[name]
    monkeypatch.setattr(parallel_evaluation, "_worker", _evaluation_failure_worker)
    before = {p.pid for p in multiprocessing.active_children()}
    prepared = (None, None, None, SimpleNamespace(component_items=lambda: ()))
    with pytest.raises(expected, match="injected worker failure"):
        list(
            parallel_evaluation.case_results(
                {"output": str(tmp_path), "error_type": name},
                [{"index": 0}],
                prepared,
                workers=4,
                on_progress=lambda *_: None,
            )
        )
    assert {p.pid for p in multiprocessing.active_children()} <= before


def test_evaluation_worker_validation_error_retains_value_error_category(monkeypatch):
    from threading import Event

    from smartsom.config import base_diagnostic
    from smartsom.config.launch import Concurrency
    from smartsom.experiments.parallel_evaluation import _worker

    class Connection:
        def __init__(self):
            self.messages = []
            self.closed = False

        def send(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    def invalid_inputs(_):
        return Concurrency(evaluation_processes=0)

    monkeypatch.setattr(base_diagnostic, "load_inputs", invalid_inputs)
    connection = Connection()
    _worker(connection, Event(), {"experiment": "invalid"})
    assert connection.closed
    kind, category, detail = connection.messages[0]
    assert (kind, category) == ("error", "ValueError")
    assert "ValidationError" in detail


@pytest.mark.parametrize(
    "authored,execution,expected",
    [(None, None, 1), (4, None, 4), (4, 1, 1), (1, 4, 4)],
)
def test_native_worker_resources_read_only_effective_operations(
    tmp_path, authored, execution, expected
):
    from smartsom.experiments.runtime_workers import NativeScheduler

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    document = "run:\n  seed: 4103\n"
    if authored is not None:
        document += f"concurrency:\n  evaluation_processes: {authored}\n"
    (inputs / "experiment.yaml").write_text(document)
    if execution is not None:
        (tmp_path / "execution.json").write_text(
            json.dumps(
                {"settings": {"concurrency": {"evaluation_processes": execution}}}
            )
        )
    job = {"root": str(tmp_path)}
    NativeScheduler._requirements(None, job)
    assert job["cpus"] == job["authored_cpus"] == 1
    assert job["evaluation_processes"] == expected
    # A stopped appended job may carry an obsolete whole-run envelope.
    job.update(cpus=20, authored_cpus=20, resource_memory=21 * 256 * 1024**2)
    NativeScheduler._requirements(None, job)
    assert job["cpus"] == 1
    assert "resource_memory" not in job


def test_native_worker_resources_reject_invalid_process_count(tmp_path):
    from smartsom.experiments.runtime_workers import NativeScheduler

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "experiment.yaml").write_text(
        "concurrency:\n  evaluation_processes: true\n"
    )
    with pytest.raises(ValueError):
        NativeScheduler._requirements(None, {"root": str(tmp_path)})


def _resizing_case_worker(connection, cancelled, settings):
    import os
    import time

    while not cancelled.is_set():
        if not connection.poll(0.02):
            continue
        task = connection.recv()
        if task is None:
            break
        time.sleep(0.04 if task["index"] % 3 == 0 else 0.01)
        connection.send(("result", task["index"], (task["index"], os.getpid())))
    connection.close()


def test_case_pool_resizes_without_reordering_or_losing_cases(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from smartsom.experiments import parallel_evaluation

    monkeypatch.setattr(parallel_evaluation, "_worker", _resizing_case_worker)
    received = []
    live_counts = []

    def capacity(live):
        live_counts.append(live)
        return 2 if len(received) < 2 else 4 if len(received) < 7 else 1

    prepared = (None, None, None, SimpleNamespace(component_items=lambda: []))
    for index, result in parallel_evaluation.case_results(
        {"output": tmp_path},
        [{"index": i} for i in range(16)],
        prepared,
        workers=2,
        on_progress=lambda *_: None,
        capacity=capacity,
    ):
        received.append((index, result))
    assert [i for i, _ in received] == list(range(16))
    assert [result[0] for _, result in received] == list(range(16))
    assert max(live_counts) == 4
    assert live_counts[-1] == 1
    import multiprocessing

    assert not {result[1] for _, result in received} & {
        p.pid for p in multiprocessing.active_children()
    }
