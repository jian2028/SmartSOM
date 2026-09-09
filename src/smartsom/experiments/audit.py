"""General physical and public-observation audits of completed or failed runs."""

import json
from collections import Counter
from pathlib import Path

import yaml

from smartsom.algorithms import ScriptedPolicy
from smartsom.config.codec import digest, primitive, read_model
from smartsom.config.models import ExecutionScheduleFile
from smartsom.config.snapshots import resolved_from_data
from smartsom.config.training import episode_input
from smartsom.dispatch import Dispatch, Transfer, Transport, WaitNextEvent, WaitUntil
from smartsom.domain import TransportDestination
from smartsom.engine import (
    DeadlockError,
    SimulationResult,
    Simulator,
    replay,
    replay_schedule,
)
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.experiments.evidence import artifact_digests
from smartsom.experiments.packaging import locate_reference
from smartsom.learning.joint_replay import replay_joint


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def actions_from_trace(trace):
    """Decode recorded semantic actions; never reconstruct them from slot numbers."""
    actions = []
    for row in trace:
        if row["kind"] == "dispatch":
            actions.append(Dispatch(**row["action"]))
        elif row["kind"] == "wait":
            action = row["action"]
            actions.append(
                WaitUntil(action["until"]) if "until" in action else WaitNextEvent()
            )
        elif row["kind"] in ("empty_start", "transfer"):
            trip = row["trip"] if row["kind"] == "empty_start" else row["transfer"]
            destination = TransportDestination(**trip["destination"])
            actions.append(
                Transport(trip["agv_id"], trip["job_id"], destination)
                if row["kind"] == "empty_start"
                else Transfer(trip["job_id"], destination)
            )
    return tuple(actions)


class _Observed:
    def __init__(self, policy):
        self.policy = policy
        self.observations = []
        self.physical_observations = []

    def select_action(self, context):
        self.observations.append(
            {"decision_index": len(self.observations), "sha256": digest(context)}
        )
        action = self.policy.select_action(context)
        if not isinstance(action, (WaitUntil, WaitNextEvent)):
            self.physical_observations.append((primitive(action), digest(context)))
        return action


def _physical(trace):
    return [
        {key: value for key, value in row.items() if key != "sequence"}
        for row in trace
        if row["kind"] not in ("decision", "wait")
    ]


def _observations(directory):
    path = directory / "observation_hashes.jsonl"
    if path.exists():
        return _rows(path)
    return [
        {"decision_index": index, "sha256": digest(row)}
        for index, row in enumerate(_rows(directory / "observations.jsonl"))
    ]


def _audit_prefix(inp, actions, trace, observations):
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    observed = []
    for index, action in enumerate(actions):
        _require(
            sim.current_decision is not None, "failed trace continues after completion"
        )
        observed.append(
            {"decision_index": len(observed), "sha256": digest(sim.current_decision)}
        )
        try:
            sim.step(action)
        except DeadlockError:
            _require(index == len(actions) - 1, "failed trace continues after deadlock")
    # A failed policy request is observed before an action can be submitted.
    if len(observations) == len(observed) + 1 and sim.current_decision is not None:
        observed.append(
            {"decision_index": len(observed), "sha256": digest(sim.current_decision)}
        )
    _require(primitive(sim.trace) == trace, "failed action-prefix trace mismatch")
    _require(observed == observations, "failed action-prefix observations mismatch")


def _audit_single(directory: str | Path) -> dict:
    """Replay retained input evidence; a failed prefix is never a complete schedule."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    _require(
        artifact_digests(directory) == manifest["artifacts"],
        "run evidence digest mismatch",
    )
    _require(
        manifest["status"] in ("completed", "failed"), "run evidence is unfinished"
    )
    failure = (
        json.loads((directory / "failure.json").read_text())
        if manifest["status"] == "failed"
        else None
    )
    snapshot_path = directory / "resolved_run.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text())
    algorithm = snapshot["algorithm"]["algorithm"]
    if algorithm.get("checkpoint"):
        algorithm["checkpoint"] = str(
            locate_reference(snapshot_path, algorithm["checkpoint"])
        )
    resolved = resolved_from_data(snapshot)
    inp = episode_input(resolved)
    trace = _rows(directory / "trace.jsonl")
    observations = _observations(directory)
    summary = json.loads((directory / "summary.json").read_text())
    actions = actions_from_trace(trace)
    complete = manifest["status"] == "completed"
    checks = []
    if complete:
        result = replay(inp.factory, inp.workload, actions, **inp.options())
        _require(primitive(result.trace) == trace, "action replay trace mismatch")
        _require(
            result.makespan == summary["makespan"], "action replay makespan mismatch"
        )
        _require(
            primitive(result.quality) == summary.get("quality"),
            "quality result mismatch",
        )
        expected_ops = {op.operation_id for op in inp.workload.operations}
        _require(
            Counter(
                row["action"]["operation_id"]
                for row in trace
                if row["kind"] == "complete"
            )
            == Counter(expected_ops)
            and summary["completed_operations"] == len(expected_ops),
            "completed operation coverage mismatch",
        )
        logistics = inp.transport_enabled or inp.buffers_enabled
        if logistics:
            schedule = read_model(
                directory / "execution_schedule.json", ExecutionScheduleFile
            )[0].execution_schedule
            expected_jobs = {
                job.job_id for order in inp.workload.orders for job in order.jobs
            }
            output_jobs = [
                move.job_id
                for move in (*result.transport_schedule, *result.transfer_schedule)
                if move.destination.kind == "output"
            ]
            _require(
                Counter(output_jobs) == Counter(expected_jobs),
                "output delivery coverage mismatch",
            )
            _require(
                summary["delivered_jobs"] == len(expected_jobs),
                "delivered job count mismatch",
            )
            _require(
                schedule == result.execution_schedule,
                "saved execution schedule mismatch",
            )
        else:
            schedule = result.schedule
        scripted = _Observed(ScriptedPolicy(actions))
        scripted_result = Simulator(inp.factory, inp.workload, **inp.options()).run(
            scripted
        )
        scripted.policy.ensure_exhausted()
        _require(scripted_result == result, "observed action replay mismatch")
        _require(
            scripted.observations == observations, "action observation replay mismatch"
        )
        replayed = replay_schedule(inp.factory, inp.workload, schedule, **inp.options())
        schedule_options = {
            key: value
            for key, value in inp.options().items()
            if key not in ("decision_trigger", "quality_probability_visibility")
        }
        scheduled = _Observed(
            ScheduleReplayPolicy(
                inp.factory, inp.workload, schedule, **schedule_options
            )
        )
        scheduled_result = Simulator(inp.factory, inp.workload, **inp.options()).run(
            scheduled
        )
        scheduled.policy.verify_result(scheduled_result)
        for candidate in (replayed, scheduled_result):
            _require(
                candidate.execution_schedule == result.execution_schedule
                and candidate.makespan == result.makespan
                and candidate.quality == result.quality
                and _physical(primitive(candidate.trace)) == _physical(trace),
                "schedule replay physical result mismatch",
            )
        # Schedule replay may replace multiple waits with a single WaitUntil.
        _require(
            scheduled.physical_observations == scripted.physical_observations,
            "schedule replay public physical-decision observations mismatch",
        )
        checks.extend(
            (
                "action_replay",
                "schedule_replay",
                "action_observations",
                "schedule_observations",
                "quality",
                "coverage",
            )
        )
    else:
        _require(
            summary.get("makespan") is None, "failed run reports a successful makespan"
        )
        if failure["stage"] in ("initialization", "solving", "schedule_validation"):
            _require(
                not trace and not observations,
                "pre-execution failure has simulation evidence",
            )
            checks.append("artifact_integrity_without_execution")
        else:
            _audit_prefix(inp, actions, trace, observations)
            checks.extend(("action_prefix", "action_observations"))
    joint = None
    if resolved.algorithm.algorithm.provider == "rllib.resource_ppo":
        audited = replay_joint(
            inp,
            resolved.algorithm.algorithm.projection,
            _rows(directory / "joint_decisions.jsonl"),
            limits=resolved.run.budget.limits(),
        )
        _require(primitive(audited.trace) == trace, "joint replay trace mismatch")
        if complete:
            _require(
                isinstance(audited.result, SimulationResult)
                and audited.result == result,
                "joint replay result mismatch",
            )
        else:
            expected_reason = next(
                (
                    reason
                    for reason in ("policy_stalled", "budget_exhausted", "deadlock")
                    if reason in failure["message"].lower()
                    or reason == "deadlock"
                    and failure["exception_type"].endswith("DeadlockError")
                ),
                None,
            )
            if expected_reason is not None:
                _require(
                    audited.reason == expected_reason,
                    "joint replay failure reason mismatch",
                )
        joint = {
            "status": "passed" if complete else "partial_verified",
            "reason": audited.reason,
            "joint_rounds": audited.rounds,
        }
        checks.append("joint_replay" if complete else "joint_prefix")
    return {
        "status": "passed" if complete else "partial_verified",
        "checks": checks,
        "joint_replay": joint,
        "schedule_replay": "passed" if complete else "not_applicable_incomplete_run",
        "trace_sha256": digest(trace),
        "observation_sha256": digest(observations),
        "world_sha256": digest(inp),
        "provider": resolved.algorithm.algorithm.provider,
        "checkpoint_sha256": getattr(
            resolved.algorithm.algorithm, "checkpoint_sha256", None
        ),
        "makespan": summary.get("makespan") if complete else None,
        "run_status": manifest["status"],
    }


def audit_run(directory: str | Path) -> dict:
    """Read-only audit of one run, an evaluation, or its outer experiment."""
    return _audit_tree(Path(directory).expanduser().resolve(), set())


def _audit_tree(directory, seen):
    _require(directory not in seen, "cyclic evaluation audit reference")
    seen.add(directory)
    record_path = directory / "run.json"
    if not record_path.exists():
        return _audit_single(directory)
    record = json.loads(record_path.read_text())
    schema = record.get("schema")
    if schema == "smartsom.experiment/v2":
        target = record.get("paths", {}).get("evaluation")
        _require(target, "experiment has no evaluation evidence to audit")
        result = _audit_tree(locate_reference(record_path, target), seen)
        return {**result, "experiment": str(directory)}
    _require(schema == "smartsom.evaluation/v1", "unsupported evaluation audit schema")
    rows = record.get("results", [])
    plan_path = locate_reference(
        record_path, record.get("paths", {}).get("plan", "plan.json")
    )
    plan = json.loads(plan_path.read_text())["entries"]
    _require(
        record.get("status") in ("completed", "completed_with_failures", "failed"),
        "evaluation is unfinished",
    )
    _require(
        len(rows) == len(plan) == record.get("requested") and len(rows) > 0,
        "evaluation plan/result coverage mismatch",
    )
    keys = [(row["case_id"], row["replication"], row["algorithm_id"]) for row in plan]
    _require(len(set(keys)) == len(keys), "duplicate evaluation plan identity")
    identities = {(case_id, algorithm_id) for case_id, _, algorithm_id in keys}
    _require(
        len(identities)
        == len(record["cases"]) * (1 + len(record["options"]["baselines"])),
        "evaluation algorithm coverage mismatch",
    )
    expected = {
        (case_id, replication, algorithm_id)
        for case_id, algorithm_id in identities
        for replication in range(record["options"]["replications"])
    }
    _require(set(keys) == expected, "evaluation replication coverage mismatch")
    _require(
        {key[0] for key in keys} == {row["case_id"] for row in record["cases"]},
        "evaluation case mapping mismatch",
    )
    paths, audited_rows = set(), []
    for planned, row in zip(plan, rows, strict=True):
        _require(
            all(row.get(key) == value for key, value in planned.items()),
            "evaluation planned identity mismatch",
        )
        try:
            _require(row.get("run_dir"), "evaluation result has no run evidence")
            child = locate_reference(record_path, row["run_dir"])
            _require(child not in paths, "duplicate evaluation run directory")
            paths.add(child)
            result = _audit_single(child)
            _require(
                result["world_sha256"] == row["world_sha256"]
                and result["provider"] == row["provider"],
                "evaluation run input/provider mismatch",
            )
            expected_checkpoint = (
                row["checkpoint"]["manifest_sha256"] if row.get("checkpoint") else None
            )
            _require(
                result["checkpoint_sha256"] == expected_checkpoint,
                "evaluation checkpoint identity mismatch",
            )
            if row["status"] == "completed":
                _require(
                    result["run_status"] == "completed"
                    and result["makespan"] == row["makespan"],
                    "evaluation completed result mismatch",
                )
            else:
                _require(
                    row.get("makespan") is None,
                    "incomplete evaluation result has makespan",
                )
                _require(
                    result["run_status"] == "failed",
                    "evaluation failure disagrees with run status",
                )
            if row.get("engineering_failure") or row["status"] == "failed":
                result = {
                    **result,
                    "status": "failed",
                    "error": "recorded engineering failure",
                }
            audited_rows.append({"run_dir": row["run_dir"], **result})
        except (OSError, ValueError, KeyError) as exc:
            audited_rows.append(
                {"run_dir": row.get("run_dir"), "status": "failed", "error": str(exc)}
            )
    completed = sum(row["status"] == "completed" for row in rows)
    _require(
        record.get("completed") == completed
        and record.get("failed") == len(rows) - completed,
        "evaluation summary counts mismatch",
    )
    _require(
        record.get("engineering_failures")
        == sum(row.get("engineering_failure", False) for row in rows),
        "evaluation engineering failure count mismatch",
    )
    expected_status = (
        "failed"
        if record["engineering_failures"]
        else "completed_with_failures"
        if record["failed"]
        else "completed"
    )
    _require(
        record["status"] == expected_status and record.get("pending") == 0,
        "evaluation final status mismatch",
    )
    summary = json.loads(
        locate_reference(record_path, record["paths"]["summary"]).read_text()
    )
    _require(
        all(
            summary.get(key) == record.get(key)
            for key in (
                "status",
                "completed",
                "failed",
                "engineering_failures",
                "pending",
                "aggregates",
            )
        ),
        "evaluation summary artifact mismatch",
    )
    statuses = {row["status"] for row in audited_rows}
    status = (
        "failed"
        if "failed" in statuses or record["status"] == "failed"
        else "partial_verified"
        if "partial_verified" in statuses
        else "passed"
    )
    return {
        "status": status,
        "evaluation": str(directory),
        "requested": len(plan),
        "verified": sum(
            row["status"] in ("passed", "partial_verified") for row in audited_rows
        ),
        "completed": completed,
        "failed": len(rows) - completed,
        "results": audited_rows,
    }
