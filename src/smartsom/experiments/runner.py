"""Single-run orchestration; all simulation transitions remain in step()."""

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from smartsom.algorithms import FirstFeasiblePolicy, ScriptedPolicy
from smartsom.config import ResolvedRun
from smartsom.config.codec import primitive
from smartsom.config.models import InstanceFile, ScriptedAlgorithm
from smartsom.engine import SimulationResult, Simulator
from smartsom.experiments.evidence import (
    append_json,
    artifact_digests,
    source_identity,
    write_json,
)
from smartsom.trace import CompletionRecord


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    simulation_result: SimulationResult


class RunFailedError(RuntimeError):
    def __init__(self, run_dir: Path, cause: BaseException):
        self.run_dir = run_dir
        self.cause = cause
        super().__init__(f"run failed in {run_dir}: {cause}")


def _policy(resolved: ResolvedRun) -> ScriptedPolicy | FirstFeasiblePolicy:
    spec = resolved.algorithm.algorithm
    if isinstance(spec, ScriptedAlgorithm):
        return ScriptedPolicy(spec.parameters.actions)
    return FirstFeasiblePolicy()


def run_one(resolved_run: ResolvedRun) -> RunResult:
    if not isinstance(resolved_run, ResolvedRun):
        raise TypeError("run_one accepts only ResolvedRun")
    resolved = resolved_run
    root = Path(resolved.run.output_root)
    root.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
    run_dir = root / attempt
    run_dir.mkdir()  # Never overwrite an earlier attempt.
    stage = "initialization"
    simulator = None
    trace_cursor = 0
    completed = 0
    last_time = 0
    manifest = {
        "schema": "smartsom.manifest/v1",
        "status": "running",
        "source": None,
        "sources": primitive(resolved.sources),
        "seed_version": resolved.seed_version,
        "seeds": primitive(resolved.seeds),
        "factory_sha256": resolved.factory_sha256,
        "workload_sha256": resolved.workload_sha256,
        "generation_provenance": primitive(resolved.provenance),
        "provider": resolved.algorithm.algorithm.provider,
        "information_projection": "decision_context",
        "artifacts": {},
    }
    try:
        with ExitStack() as stack:
            progress = stack.enter_context(
                (run_dir / "progress.log").open("x", encoding="utf-8", buffering=1)
            )
            progress.write("initializing\n")
            write_json(run_dir / "manifest.json", manifest)
            (run_dir / "resolved_run.yaml").write_text(
                yaml.safe_dump(
                    {"schema": "smartsom.resolved-run/v1", **primitive(resolved)},
                    sort_keys=True,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            instance = InstanceFile(
                schema="smartsom.workload-instance/v1",
                workload=resolved.workload,
                content_sha256=resolved.workload_sha256,
                provenance=resolved.provenance,
            )
            write_json(run_dir / "realized_instance.json", instance)
            manifest["source"] = source_identity()
            policy = _policy(resolved)
            manifest["provider_implementation"] = (
                f"{type(policy).__module__}.{type(policy).__qualname__}"
            )
            write_json(run_dir / "manifest.json", manifest)
            stage = "simulation"
            trace_file = stack.enter_context(
                (run_dir / "trace.jsonl").open("x", encoding="utf-8")
            )
            metrics_file = stack.enter_context(
                (run_dir / "metrics.jsonl").open("x", encoding="utf-8")
            )

            def drain():
                nonlocal trace_cursor, completed, last_time
                for record in simulator.trace[trace_cursor:]:
                    append_json(trace_file, record)
                    trace_cursor += 1
                    last_time = record.simulation_time
                    if isinstance(record, CompletionRecord):
                        completed += 1
                        append_json(
                            metrics_file,
                            {
                                "kind": "completion",
                                "simulation_time": last_time,
                                "completed_operations": completed,
                            },
                        )

            simulator = Simulator(resolved.factory, resolved.workload)
            drain()
            context = simulator.current_decision
            while context is not None:
                try:
                    outcome = simulator.step(policy.select_action(context))
                finally:
                    drain()
                progress.write(f"tick={last_time} completed_operations={completed}\n")
                if isinstance(outcome, SimulationResult):
                    result = outcome
                    break
                context = outcome
            if isinstance(policy, ScriptedPolicy):
                policy.ensure_exhausted()
            stage = "finalization"
            summary = {
                "schema": "smartsom.summary/v1",
                "status": "completed",
                "end_reason": result.end_reason,
                "simulation_time": result.makespan,
                "completed_operations": completed,
                "makespan": result.makespan,
            }
            append_json(metrics_file, {"kind": "terminal", **summary})
            write_json(run_dir / "summary.json", summary)
            progress.write(f"completed makespan={result.makespan}\n")
        manifest.update(status="completed", artifacts=artifact_digests(run_dir))
        write_json(run_dir / "manifest.json", manifest)
        return RunResult(run_dir, result)
    except (Exception, KeyboardInterrupt) as exc:
        # Use actual trace even if an output writer failed while draining it.
        if simulator is not None:
            records = simulator.trace
            completed = sum(isinstance(record, CompletionRecord) for record in records)
            last_time = records[-1].simulation_time if records else 0
        failure = {
            "schema": "smartsom.failure/v1",
            "stage": stage,
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "simulation_time": last_time,
        }
        if hasattr(exc, "action"):
            failure["action"] = primitive(exc.action)
        try:
            write_json(run_dir / "failure.json", failure)
            write_json(
                run_dir / "summary.json",
                {
                    "schema": "smartsom.summary/v1",
                    "status": "failed",
                    "end_reason": "execution_failed",
                    "simulation_time": last_time,
                    "completed_operations": completed,
                    "makespan": None,
                },
            )
            with (run_dir / "progress.log").open("a", encoding="utf-8") as progress:
                progress.write(f"failed stage={stage}: {exc}\n")
            manifest.update(status="failed", artifacts=artifact_digests(run_dir))
            write_json(run_dir / "manifest.json", manifest)
        except OSError as evidence_error:
            exc.add_note(f"Could not finish failure evidence: {evidence_error}")
        raise RunFailedError(run_dir, exc) from exc
