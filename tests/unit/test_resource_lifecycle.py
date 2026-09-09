"""Independent lifecycle boundaries shared by Parallel and checkpoint execution."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_buffers import dispatch, move, vehicle_case
from test_resource_projection import indices
from test_resource_training import bundle, inject_predictor
from test_static_engine import crossing_case, op, problem

from smartsom.config import resolve_run
from smartsom.config.codec import digest, primitive
from smartsom.config.models import EpisodeBudget
from smartsom.domain import ArrivalPlan, Job, JobArrival, MachineBuffers
from smartsom.engine import DeadlockError, Simulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.evidence import write_json
from smartsom.learning.checkpoint import file_hash, structural_identity
from smartsom.learning.episode import EpisodeInput
from smartsom.learning.joint import PolicyStalledError
from smartsom.learning.joint_evidence import step_record
from smartsom.learning.joint_replay import replay_joint
from smartsom.learning.resources import ResourceProjection

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def fixture_checkpoint_backend(monkeypatch):
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda p: {})


@pytest.fixture
def parallel_env_class():
    if os.environ.get("SMARTSOM_REQUIRE_PETTINGZOO") == "1":
        __import__("pettingzoo")
    else:
        pytest.importorskip("pettingzoo")
    from smartsom.learning.pettingzoo import SmartSOMParallelEnv

    return SmartSOMParallelEnv


def configured_case(tmp_path, inp, budget):
    """Bind the existing non-model checkpoint fixture to a hand-sized input."""
    template, checkpoint = bundle(tmp_path)
    base = resolve_run(ROOT / "configs/runs/crossing.yaml")
    scenario = base.scenario.model_copy(
        update={
            "arrivals": template.scenario.arrivals if inp.arrivals else None,
            "transport": template.scenario.transport if inp.transport_enabled else None,
            "buffers": template.scenario.buffers if inp.buffers_enabled else None,
        }
    )
    resolved = replace(
        base,
        factory=inp.factory,
        workload=inp.workload,
        factory_sha256=digest(inp.factory),
        workload_sha256=digest(inp.workload),
        provenance=None,
        scenario=scenario,
        algorithm=template.algorithm,
        arrivals=inp.arrivals,
        arrivals_sha256=digest(inp.arrivals) if inp.arrivals else None,
        transport_enabled=inp.transport_enabled,
        transport_sha256=digest(inp.factory.transport)
        if inp.transport_enabled
        else None,
        buffers_enabled=inp.buffers_enabled,
        buffers_sha256=digest(inp.factory.buffers) if inp.buffers_enabled else None,
        run=base.run.model_copy(
            update={"budget": budget, "output_root": str(tmp_path / "runs")}
        ),
    )
    projection = ResourceProjection(
        inp.factory,
        resolved.algorithm.algorithm.projection,
        transport_enabled=inp.transport_enabled,
    )
    mapping = [
        [a, "machine_policy" if a.startswith("machine:") else "agv_policy"]
        for a in projection.agents
    ]
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    weights = [
        w for w in metadata["role_weights"] if w["role"] in {r for _, r in mapping}
    ]
    metadata.update(
        structure_sha256=structural_identity(resolved, projection.spec),
        role_mapping=mapping,
        role_weights=weights,
        agent_steps=metadata["environment_steps"] * len(mapping),
        initial_weights_sha256=digest(
            {w["role"]: w["initial_sha256"] for w in weights}
        ),
        final_weights_sha256=digest({w["role"]: w["final_sha256"] for w in weights}),
    )
    write_json(checkpoint / "checkpoint.json", metadata)
    return replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={
                "algorithm": resolved.algorithm.algorithm.model_copy(
                    update={
                        "checkpoint_sha256": file_hash(checkpoint / "checkpoint.json")
                    }
                )
            }
        ),
    )


def scripted_proposals(rounds):
    proposals = iter(rounds)
    return lambda decision: indices(decision, *next(proposals))


def lifecycle_case(name):
    if name == "deadlock":
        factory, workload, _ = vehicle_case()
        factory = replace(
            factory,
            buffers=(MachineBuffers("M1", 0, 0),),
            transport=replace(factory.transport, agvs=factory.transport.agvs[:1]),
        )
        inp = EpisodeInput(
            factory, workload, transport_enabled=True, buffers_enabled=True
        )
        rounds = ((move("A", "M1", "V1"),), (dispatch("A"),), (move("B", "M1", "V1"),))
        return inp, EpisodeBudget(), rounds, "deadlock", -10001, 6
    if name == "round_limit":
        factory, workload, _ = crossing_case()
        rounds = ((dispatch("C1"), dispatch("D1")),)
        return (
            EpisodeInput(factory, workload),
            EpisodeBudget(max_decisions=1),
            rounds,
            "budget_exhausted",
            -10001,
            3,
        )
    jobs = [Job("J", (op("J1", "M1", 3),))]
    if name == "exact_tick_nonterminal":
        jobs.append(Job("K", (op("K1", "M1", 2),)))
    factory, workload = problem(*jobs)
    inp = EpisodeInput(
        factory,
        workload,
        arrivals=ArrivalPlan(tuple(JobArrival(j.job_id, 5, 5) for j in jobs)),
    )
    if name == "policy_stall":
        return inp, EpisodeBudget(), ((),), "policy_stalled", -10001, 5
    limit = (
        6
        if name == "atomic_overshoot"
        else 8
        if name.startswith("exact_tick")
        else 10000
    )
    reason = (
        "budget_exhausted"
        if name in ("atomic_overshoot", "exact_tick_nonterminal")
        else "completed"
    )
    reward = -9 if name == "exact_tick_nonterminal" else -8
    return inp, EpisodeBudget(max_ticks=limit), ((dispatch("J1"),),), reason, reward, 8


@pytest.mark.pettingzoo
@pytest.mark.parametrize(
    "name",
    [
        "initial_advance",
        "exact_tick_completed",
        "exact_tick_nonterminal",
        "atomic_overshoot",
        "round_limit",
        "policy_stall",
        "deadlock",
    ],
)
def test_parallel_and_checkpoint_runner_share_lifecycle_and_exact_ledger(
    name, tmp_path, monkeypatch, parallel_env_class
):
    inp, budget, rounds, reason, reward, tick = lifecycle_case(name)
    resolved = configured_case(tmp_path, inp, budget)
    spec = resolved.algorithm.algorithm.projection
    env = parallel_env_class(inp, spec, limits=budget.limits())
    env.reset()
    agent_returns = dict.fromkeys(env.possible_agents, 0.0)
    for proposals in rounds:
        _, rewards, terminated, truncated, infos = env.step(
            indices(env.projected, *proposals)
        )
        for agent, value in rewards.items():
            agent_returns[agent] += value
    assert env.finished and not env.agents
    assert env.reason == reason and env.rewarded_tick == tick
    assert env.total_reward == reward and set(agent_returns.values()) == {reward}
    assert all(truncated.values()) == (reason == "budget_exhausted")
    assert all(terminated.values()) == (reason != "budget_exhausted")
    assert {info["end_reason"] for info in infos.values()} == {reason}

    inject_predictor(monkeypatch, scripted_proposals(rounds))
    if reason == "completed":
        result = run_one(resolved)
        run_dir = result.run_dir
        assert result.simulation_result == env.result
        assert result.simulation_result.makespan == 8
    else:
        with pytest.raises(RunFailedError) as failure:
            run_one(resolved)
        run_dir = failure.value.run_dir
        if reason == "deadlock":
            assert isinstance(failure.value.cause, DeadlockError)
            assert any(r.kind == "wait_for_unload" for r in env.simulator.trace)
        elif reason == "policy_stalled":
            assert isinstance(failure.value.cause, PolicyStalledError)
        else:
            assert "budget_exhausted" in str(failure.value.cause)
        assert json.loads((run_dir / "summary.json").read_text())["makespan"] is None

    records = [
        json.loads(line)
        for line in (run_dir / "joint_decisions.jsonl").read_text().splitlines()
    ]
    expected = primitive(
        [step_record(i, step, full=True) for i, step in enumerate(env.steps)]
    )
    assert records == expected
    assert sum(row["reward"] for row in records) == reward
    assert records[-1]["reason"] == reason
    assert records[-1]["simulation_time"] == tick
    if inp.arrivals:
        assert records[0]["start_tick"] == 5
    physical_trace = [
        json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert physical_trace == primitive(env.simulator.trace)
    audited = replay_joint(inp, spec, records, limits=budget.limits())
    assert audited.trace == env.simulator.trace
    assert audited.reason == reason and audited.total_reward == reward
    assert audited.actions == tuple(
        action for step in env.steps for action in step.actions
    )
    assert audited.result == (env.result if reason == "completed" else None)


def test_joint_writer_failure_after_physical_completion_preserves_trace_and_cause(
    tmp_path, monkeypatch
):
    factory, workload = problem(Job("J", (op("J1", "M1", 3),)))
    inp = EpisodeInput(factory, workload)
    resolved = configured_case(tmp_path, inp, EpisodeBudget())
    action = dispatch("J1")
    direct = Simulator(factory, workload)
    completed = direct.step(action)
    inject_predictor(monkeypatch, scripted_proposals(((action,),)))
    original = OSError("joint writer failed after physical completion")
    attempted = []

    def fail_joint_write(stream, row):
        attempted.append(row)
        raise original

    monkeypatch.setattr("smartsom.experiments.runner.append_json", fail_joint_write)
    with pytest.raises(RunFailedError) as failure:
        run_one(resolved)
    assert failure.value.cause is original
    assert len(attempted) == 1
    assert attempted[0]["actions"] == (action,)
    assert attempted[0]["reason"] == "completed"
    assert attempted[0]["simulation_time"] == completed.makespan == 3
    run_dir = failure.value.run_dir
    trace = [
        json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert trace == primitive(direct.trace)
    assert {row["kind"] for row in trace} >= {"dispatch", "complete", "terminate"}
    assert (run_dir / "joint_decisions.jsonl").read_text() == ""
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((run_dir / "summary.json").read_text())["makespan"] is None
    recorded_failure = json.loads((run_dir / "failure.json").read_text())
    assert recorded_failure["exception_type"] == "builtins.OSError"
    assert recorded_failure["message"] == str(original)
