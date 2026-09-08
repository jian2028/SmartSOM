"""Independent run-evidence audit using public replay and decision interfaces."""

import json
import time
from collections import Counter
from pathlib import Path

from smartsom.algorithms import ScriptedPolicy
from smartsom.config import load_resolved_run
from smartsom.config.codec import digest, primitive, read_model
from smartsom.config.models import ExecutionScheduleFile
from smartsom.dispatch import Dispatch, Transfer, Transport, WaitNextEvent, WaitUntil
from smartsom.domain import TransportDestination
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.experiments.evidence import artifact_digests


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def actions_from_trace(trace):
    actions = []
    for row in trace:
        kind = row["kind"]
        if kind == "dispatch":
            actions.append(Dispatch(**row["action"]))
        elif kind == "wait":
            action = row["action"]
            actions.append(
                WaitUntil(action["until"]) if "until" in action else WaitNextEvent()
            )
        elif kind == "empty_start":
            trip = row["trip"]
            actions.append(
                Transport(
                    trip["agv_id"],
                    trip["job_id"],
                    TransportDestination(**trip["destination"]),
                )
            )
        elif kind == "transfer":
            trip = row["transfer"]
            actions.append(
                Transfer(trip["job_id"], TransportDestination(**trip["destination"]))
            )
    return tuple(actions)


def simulator_options(resolved):
    return dict(
        arrivals=resolved.arrivals,
        decision_trigger=resolved.scenario.decision_trigger,
        processing_times=resolved.processing_times,
        machine_events=resolved.machine_events,
        transport_enabled=resolved.transport_enabled,
        buffers_enabled=resolved.buffers_enabled,
        holding_buffer_enabled=resolved.holding_buffer_enabled,
        quality=resolved.quality,
        quality_probability_visibility=resolved.scenario.quality.probability_visibility
        if resolved.scenario.quality
        else "public",
    )


class ObservedPolicy:
    """Capture exactly the public context presented before each submitted action."""

    def __init__(self, policy):
        self.policy = policy
        self.observations = []

    def select_action(self, context):
        self.observations.append(
            {"decision_index": len(self.observations), "sha256": digest(context)}
        )
        return self.policy.select_action(context)


def audit_run(run_dir: Path, *, expected_jobs: int, expected_operations: int) -> dict:
    """No input regeneration, file writes, solver, or second transition model."""
    started = time.monotonic()
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    require(manifest["status"] == "completed", "run did not complete")
    require(
        artifact_digests(run_dir) == manifest["artifacts"],
        "run evidence digest mismatch",
    )
    resolved = load_resolved_run(run_dir / "resolved_run.yaml")
    trace = read_jsonl(run_dir / "trace.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    hashes = read_jsonl(run_dir / "observation_hashes.jsonl")
    schedule = read_model(run_dir / "execution_schedule.json", ExecutionScheduleFile)[
        0
    ].execution_schedule
    actions = actions_from_trace(trace)
    options = simulator_options(resolved)
    expected_ids = {j.job_id for order in resolved.workload.orders for j in order.jobs}
    operation_ids = {op.operation_id for op in resolved.workload.operations}
    require(
        len(expected_ids) == expected_jobs
        and len(operation_ids) == expected_operations,
        "input coverage mismatch",
    )
    require(
        summary["delivered_jobs"] == expected_jobs
        and summary["completed_operations"] == expected_operations,
        "summary coverage mismatch",
    )
    require(
        Counter(r["action"]["operation_id"] for r in trace if r["kind"] == "complete")
        == Counter(operation_ids),
        "missing or duplicate completion",
    )
    require(
        Counter(
            r["trip"]["job_id"]
            for r in trace
            if r["kind"] == "delivery" and r["trip"]["destination"]["kind"] == "output"
        )
        == Counter(expected_ids),
        "missing or duplicate output delivery",
    )

    def compare(result, *, full_trace=False):
        require(
            result.execution_schedule == schedule, "replay execution schedule mismatch"
        )
        require(result.makespan == summary["makespan"], "replay makespan mismatch")
        require(
            primitive(result.quality) == summary.get("quality"),
            "replay quality mismatch",
        )
        require(
            result.makespan
            == max(
                t.delivery_time
                for t in result.transport_schedule
                if t.destination.kind == "output"
            ),
            "makespan is not final output",
        )
        actual = primitive(result.trace)
        if full_trace:
            require(actual == trace, "action replay trace mismatch")
        else:
            # WaitUntil may replace repeated WaitNextEvent. Physical/quality events
            # and their order must remain exact; trace sequence numbers may differ.
            def physical(rows):
                return [
                    {k: v for k, v in row.items() if k != "sequence"}
                    for row in rows
                    if row["kind"] not in ("decision", "wait")
                ]

            require(
                physical(actual) == physical(trace),
                "schedule replay physical or quality trace mismatch",
            )

    # Exercise both public replay APIs on every accepted run.
    result = replay(resolved.factory, resolved.workload, actions, **options)
    compare(result, full_trace=True)
    compare(replay_schedule(resolved.factory, resolved.workload, schedule, **options))

    # The policies below only read observations; all transitions still use the
    # exact Simulator.run -> step path used by the public replay APIs.
    scripted = ObservedPolicy(ScriptedPolicy(actions))
    compare(
        Simulator(resolved.factory, resolved.workload, **options).run(scripted),
        full_trace=True,
    )
    scripted.policy.ensure_exhausted()
    require(
        scripted.observations == hashes, "action replay observation digest mismatch"
    )
    schedule_options = {
        k: v
        for k, v in options.items()
        if k not in ("decision_trigger", "quality_probability_visibility")
    }
    schedule_policy = ScheduleReplayPolicy(
        resolved.factory, resolved.workload, schedule, **schedule_options
    )
    observed_schedule = ObservedPolicy(schedule_policy)
    schedule_result = Simulator(resolved.factory, resolved.workload, **options).run(
        observed_schedule
    )
    schedule_policy.verify_result(schedule_result)
    compare(schedule_result)
    require(
        observed_schedule.observations == hashes,
        "schedule replay observation digest mismatch",
    )
    if result.quality is not None:
        for name in ("passed_jobs", "defective_jobs", "total_jobs", "passing_rate"):
            require(
                summary.get(name) == getattr(result.quality, name),
                f"quality summary {name} mismatch",
            )
        require(
            Counter(q.operation_id for q in result.quality.operations)
            == Counter(operation_ids),
            "missing or duplicate operation quality check",
        )
        require(
            Counter(q.job_id for q in result.quality.jobs) == Counter(expected_ids),
            "missing or duplicate inspection",
        )
    return {
        "status": "passed",
        "run_dir": str(run_dir),
        "source": manifest["source"],
        "elapsed_seconds": time.monotonic() - started,
        "input_sha256": {
            "factory": resolved.factory_sha256,
            "workload": resolved.workload_sha256,
            "quality": resolved.quality_draws_sha256,
        },
        "makespan": result.makespan,
        "passing_rate": result.quality.passing_rate if result.quality else None,
        "passed_job_ids": sorted(q.job_id for q in result.quality.jobs if q.passed)
        if result.quality
        else None,
        "completed_operations": expected_operations,
        "delivered_jobs": expected_jobs,
        "holding_trips": sum(
            t.destination.kind == "holding" for t in result.transport_schedule
        ),
        "transports": len(result.transport_schedule),
        "actions": len(result.actions),
        "trace_sha256": digest(trace),
        "execution_schedule_sha256": digest(schedule),
        "observation_sha256": digest(hashes),
        "observed_decisions": len(hashes),
        "checks": [
            "action_replay",
            "schedule_replay",
            "action_observations",
            "schedule_observations",
            "quality",
            "coverage",
            "kernel_invariants_each_transition",
        ],
        "evidence_bytes": sum(
            p.stat().st_size for p in run_dir.iterdir() if p.is_file()
        ),
    }
