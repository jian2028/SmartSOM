"""Read-only resource ledger audits: joint coordination and both physical replays."""

import json
from pathlib import Path

from smartsom.config.codec import digest, primitive
from smartsom.config.snapshots import load_resolved_run
from smartsom.config.training import episode_input
from smartsom.engine import replay, replay_schedule
from smartsom.learning.joint_replay import replay_joint


def _physical_replays(inp, audited):
    result = audited.result
    if result is None:
        raise ValueError("joint ledger did not complete the physical simulation")
    actions = replay(inp.factory, inp.workload, audited.actions, **inp.options())
    if actions != result:
        raise ValueError("resource action replay differs from joint execution")
    schedule = (
        result.execution_schedule
        if inp.transport_enabled or inp.buffers_enabled
        else result.schedule
    )
    scheduled = replay_schedule(inp.factory, inp.workload, schedule, **inp.options())
    if (
        scheduled.schedule != result.schedule
        or scheduled.execution_schedule != result.execution_schedule
        or scheduled.quality != result.quality
        or scheduled.makespan != result.makespan
    ):
        raise ValueError("resource exact schedule replay mismatch")


def audit_resource_episode(resolved, inp, row):
    audited = replay_joint(
        inp,
        resolved.algorithm.algorithm.projection,
        row["steps"],
        limits=resolved.run.budget.limits(),
    )
    if (
        digest(audited.trace) != row["trace_sha256"]
        or digest(audited.actions) != row["actions_sha256"]
        or primitive(audited.bindings) != row["bindings"]
        or audited.total_reward != row["return"]
    ):
        raise ValueError("resource episode trace/action/binding/return mismatch")
    if row["end_reason"] == "training_budget_stop":
        if audited.reason is not None:
            raise ValueError("partial resource episode was actually terminal")
    elif audited.reason != row["end_reason"]:
        raise ValueError("resource episode termination mismatch")
    if audited.reason == "completed":
        if (
            audited.result.makespan != row["makespan"]
            or primitive(audited.result.quality) != row["quality"]
        ):
            raise ValueError("resource episode result mismatch")
        _physical_replays(inp, audited)
    elif row["makespan"] is not None or row["quality"] is not None:
        raise ValueError("failed resource episode reports successful metrics")
    return audited


def audit_joint_run(run_dir):
    run_dir = Path(run_dir)
    resolved = load_resolved_run(run_dir / "resolved_run.yaml")
    if resolved.algorithm.algorithm.provider != "rllib.resource_ppo":
        raise ValueError("joint audit requires a resource checkpoint run")
    inp = episode_input(resolved)
    records = [
        json.loads(line)
        for line in (run_dir / "joint_decisions.jsonl").read_text().splitlines()
    ]
    audited = replay_joint(
        inp,
        resolved.algorithm.algorithm.projection,
        records,
        limits=resolved.run.budget.limits(),
    )
    trace = [json.loads(s) for s in (run_dir / "trace.jsonl").read_text().splitlines()]
    if primitive(audited.trace) != trace:
        raise ValueError("joint evaluation trace mismatch")
    _physical_replays(inp, audited)
    return {
        "status": "passed",
        "joint_rounds": audited.rounds,
        "agent_steps": sum(len(r["indices"]) for r in records),
        "physical_actions": len(audited.actions),
        "conflicts": sum(
            p["disposition"] in ("job_claimed", "no_longer_feasible")
            for r in records
            for p in r["proposals"]
        ),
        "team_return": audited.total_reward,
        "makespan": audited.result.makespan,
    }
