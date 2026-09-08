"""Local evidence serialization and source identity, outside the engine."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from collections.abc import Sequence
from contextlib import ExitStack
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

from smartsom.algorithms.solver import ScheduleSolution, SolveRequest, SolverStatus
from smartsom.config import ResolvedRun
from smartsom.config.arrivals import arrival_rows
from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.models import (
    ExecutionScheduleFile,
    GenerationProvenance,
    InstanceFile,
    MachineEventFile,
    ProcessingTimeFile,
    QualityFile,
)
from smartsom.dispatch import DecisionContext
from smartsom.engine.result import SimulationResult
from smartsom.experiments.progress import RunProgress
from smartsom.trace import CompletionRecord, TraceRecord, TransportRecord
from smartsom.trace.records import InspectionRecord, TransferRecord, VehicleRecord
from smartsom.workloads.fjs import ImportProvenance


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            primitive(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_json(stream, value) -> None:
    stream.write(canonical_json(value) + "\n")
    stream.flush()


def source_identity() -> dict:
    # Find the checkout that supplies this code, not the caller's working directory.
    root = next(
        (p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None
    )
    git = {"commit": None, "dirty": None, "status": None}
    if root is not None:
        try:

            def command(*args):
                return subprocess.check_output(
                    ["git", "-C", str(root), *args],
                    text=True,
                    stderr=subprocess.PIPE,
                    timeout=5,
                ).strip()

            commit = command("rev-parse", "HEAD")
            status = command("status", "--porcelain", "--untracked-files=all")
            git = {"commit": commit, "dirty": bool(status), "status": status}
        except (OSError, subprocess.SubprocessError) as exc:
            git["unavailable_reason"] = str(exc)
    versions = {}
    for package in (
        "smartsom",
        "pydantic",
        "pydantic-core",
        "PyYAML",
        "annotated-types",
        "typing-extensions",
        "typing-inspection",
        "pyjobshop",
        "ortools",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git": git,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def artifact_digests(run_dir: Path) -> dict[str, str]:
    # Manifest excludes itself, avoiding a recursive self-checksum.
    return {
        path.name: _file_digest(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
        and path.name != "manifest.json"
        and not path.name.endswith(".tmp")
    }


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RunEvidence:
    """Write one attempt's evidence; never own or advance simulation state."""

    def __init__(self, run_dir: Path, resolved: ResolvedRun, on_progress=None):
        self.on_progress = on_progress
        self.started = time.monotonic()
        self.last_notification = 0.0
        self.observation_cursor = 0
        self.debug_logger = None
        self.run_dir = run_dir
        self.resolved = resolved
        self.trace_cursor = 0
        self.completed = 0
        self.delivered = 0
        self.inspected = 0
        self.passed = 0
        self.last_time = 0
        self.manifest = {
            "schema": "smartsom.manifest/v1",
            "status": "running",
            "source": None,
            "sources": primitive(resolved.sources),
            "seed_version": resolved.seed_version,
            "seeds": primitive(resolved.seeds),
            "factory_sha256": resolved.factory_sha256,
            "buffers_enabled": resolved.buffers_enabled,
            "buffers_sha256": resolved.buffers_sha256,
            "buffer_capacities": primitive(resolved.factory.buffers)
            if resolved.buffers_enabled
            else None,
            "buffer_admission_rule": "immediate_capacity"
            if resolved.buffers_enabled
            and resolved.algorithm.algorithm.provider
            in ("builtin.spt", "builtin.first_feasible")
            else None,
            "transport_enabled": resolved.transport_enabled,
            "transport_sha256": resolved.transport_sha256,
            "workload_sha256": resolved.workload_sha256,
            "arrivals_sha256": resolved.arrivals_sha256,
            "processing_times_sha256": resolved.processing_times_sha256,
            "machine_events_sha256": resolved.machine_events_sha256,
            "machine_event_provenance": primitive(resolved.machine_event_provenance),
            "processing_provenance": primitive(resolved.processing_provenance),
            "arrival_provenance": primitive(resolved.arrival_provenance),
            "decision_trigger": resolved.scenario.decision_trigger,
            "generation_provenance": primitive(resolved.provenance)
            if isinstance(resolved.provenance, GenerationProvenance)
            else None,
            "import_provenance": primitive(resolved.provenance)
            if isinstance(resolved.provenance, ImportProvenance)
            else None,
            "provider": resolved.algorithm.algorithm.provider,
            "information_projection": resolved.algorithm.algorithm.required_information,
            "scenario_visibility": resolved.scenario.visibility,
            "interface_kind": resolved.algorithm.algorithm.interface_kind,
            "artifacts": {},
        }

        if resolved.study_seed_origin is not None:
            self.manifest["study_seed_origin"] = primitive(resolved.study_seed_origin)
        if resolved.run.recording is not None:
            self.manifest["recording"] = primitive(resolved.run.recording)
        if resolved.quality is not None:
            self.manifest.update(
                quality_draws_sha256=resolved.quality_draws_sha256,
                quality_modes_sha256=resolved.quality_modes_sha256,
                quality_provenance=primitive(resolved.quality_provenance),
                quality_probability_visibility=resolved.scenario.quality.probability_visibility,
                quality_mode=getattr(
                    resolved.algorithm.algorithm.parameters, "quality_mode", None
                ),
            )

    def initialize(self, stack: ExitStack) -> None:
        run_dir, resolved, manifest = self.run_dir, self.resolved, self.manifest
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
        observations = None
        self.observation_hashes = None
        if resolved.arrivals is not None:
            with (run_dir / "realized_events.jsonl").open(
                "x", encoding="utf-8"
            ) as events:
                for row in arrival_rows(resolved.arrivals, resolved.arrival_provenance):
                    append_json(events, row)
        if resolved.processing_times is not None:
            write_json(
                run_dir / "realized_processing_times.json",
                ProcessingTimeFile(
                    schema="smartsom.processing-times/v1",
                    processing_times=resolved.processing_times,
                    content_sha256=resolved.processing_times_sha256,
                    provenance=resolved.processing_provenance,
                ),
            )
        if resolved.machine_events is not None:
            write_json(
                run_dir / "realized_machine_events.json",
                MachineEventFile(
                    schema="smartsom.machine-events/v1",
                    machine_events=resolved.machine_events,
                    content_sha256=resolved.machine_events_sha256,
                    provenance=resolved.machine_event_provenance,
                ),
            )
        if resolved.quality is not None:
            write_json(
                run_dir / "realized_quality.json",
                QualityFile(
                    schema="smartsom.quality-draws/v1",
                    draws=resolved.quality.draws,
                    content_sha256=resolved.quality_draws_sha256,
                    provenance=resolved.quality_provenance,
                ),
            )
            write_json(
                run_dir / "effective_modes.json",
                {
                    "schema": "smartsom.quality-execution-modes/v1",
                    "workload_sha256": resolved.workload_sha256,
                    "processing_times_sha256": resolved.processing_times_sha256,
                    "content_sha256": resolved.quality_modes_sha256,
                    "modes": primitive(resolved.quality.modes),
                },
            )
        if (
            resolved.run.recording is not None
            or resolved.quality is not None
            or resolved.arrivals is not None
            or resolved.processing_times is not None
            or resolved.machine_events is not None
            or resolved.transport_enabled
            or resolved.buffers_enabled
        ):
            if resolved.run.recording and resolved.run.recording.observations == "hash":
                self.observation_hashes = stack.enter_context(
                    (run_dir / "observation_hashes.jsonl").open("x", encoding="utf-8")
                )
            else:
                observations = stack.enter_context(
                    (run_dir / "observations.jsonl").open("x", encoding="utf-8")
                )
        manifest["source"] = source_identity()
        self.progress = progress
        self.observations = observations
        if resolved.run.recording and resolved.run.recording.debug:
            logger = logging.Logger(str(run_dir), logging.DEBUG)
            handler = RotatingFileHandler(
                run_dir / "debug.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=2,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
            stack.callback(handler.close)
            self.debug_logger = logger
        self.notify("initialization", force=True)

    def notify(self, stage: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.debug_logger:
            self.debug_logger.debug(
                "stage=%s tick=%s trace_cursor=%s",
                stage,
                self.last_time,
                self.trace_cursor,
            )
        if self.on_progress is not None and (
            force or now - self.last_notification >= 1
        ):
            self.on_progress(
                RunProgress(
                    self.run_dir,
                    stage,
                    now - self.started,
                    self.last_time,
                    self.completed,
                    self.delivered,
                    self.inspected,
                    self.passed,
                )
            )
            self.last_notification = now

    def record_provider(self, provider) -> None:
        self.manifest["provider_implementation"] = (
            f"{type(provider).__module__}.{type(provider).__qualname__}"
        )
        write_json(self.run_dir / "manifest.json", self.manifest)

    def start_execution(self, stack: ExitStack) -> None:
        self.trace_file = stack.enter_context(
            (self.run_dir / "trace.jsonl").open("x", encoding="utf-8")
        )
        self.metrics_file = stack.enter_context(
            (self.run_dir / "metrics.jsonl").open("x", encoding="utf-8")
        )

    def record_solver_request(self, request: SolveRequest) -> None:
        self.solver_settings = {
            "solver_time_limit_seconds": request.solver_time_limit_seconds,
            "num_workers": 1,
            "solver_seed": request.solver_seed,
            "backend_seed": request.backend_seed,
        }
        self.manifest["solver_settings"] = self.solver_settings
        write_json(self.run_dir / "manifest.json", self.manifest)
        self.progress.write(f"solving provider={self.manifest['provider']}\n")
        self.notify("solving", force=True)

    def record_solver_result(self, solution: ScheduleSolution) -> None:
        write_json(
            self.run_dir / "solver_result.json",
            {
                "schema": "smartsom.solver-result/v1",
                **primitive(solution),
                "gap": solution.gap,
                **self.solver_settings,
            },
        )
        self.manifest["solver_status"] = solution.status
        write_json(self.run_dir / "manifest.json", self.manifest)
        self.progress.write(
            f"solver status={solution.status} objective={solution.objective} bound={solution.bound}\n"
        )

    def observe(self, context: DecisionContext) -> None:
        if self.observation_hashes is not None:
            append_json(
                self.observation_hashes,
                {"decision_index": self.observation_cursor, "sha256": digest(context)},
            )
        self.observation_cursor += 1
        if self.observations is not None:
            append_json(self.observations, context)

    def drain(self, records: tuple[TraceRecord, ...]) -> None:
        for record in records:
            append_json(self.trace_file, record)
            self.trace_cursor += 1
            self.last_time = record.simulation_time
            if (
                isinstance(record, (TransportRecord, VehicleRecord))
                and record.kind == "delivery"
                and record.trip.destination.kind == "output"
            ):
                self.delivered += 1
            if (
                isinstance(record, TransferRecord)
                and record.transfer.destination.kind == "output"
            ):
                self.delivered += 1
            if isinstance(record, InspectionRecord):
                self.inspected += 1
                self.passed += int(record.inspection.passed)
                append_json(
                    self.metrics_file,
                    {
                        "kind": "inspection",
                        "simulation_time": self.last_time,
                        "inspected_jobs": self.inspected,
                        "passed_jobs": self.passed,
                        "defective_jobs": self.inspected - self.passed,
                    },
                )
            if isinstance(record, CompletionRecord):
                self.completed += 1
                append_json(
                    self.metrics_file,
                    {
                        "kind": "completion",
                        "simulation_time": self.last_time,
                        "completed_operations": self.completed,
                    },
                )

    def record_progress(self) -> None:
        self.notify("simulation")
        suffix = (
            f" delivered_jobs={self.delivered}"
            if self.resolved.transport_enabled or self.resolved.buffers_enabled
            else ""
        )
        if self.resolved.quality is not None:
            suffix += f" inspected_jobs={self.inspected} passed_jobs={self.passed} defective_jobs={self.inspected - self.passed}"
        self.progress.write(
            f"tick={self.last_time} completed_operations={self.completed}{suffix}\n"
        )

    def complete(
        self, result: SimulationResult, solution: ScheduleSolution | None
    ) -> None:
        summary = {
            "schema": "smartsom.summary/v1",
            "status": "completed",
            "end_reason": result.end_reason,
            "simulation_time": result.makespan,
            "completed_operations": self.completed,
            "makespan": result.makespan,
        }
        if self.resolved.transport_enabled or self.resolved.buffers_enabled:
            summary["delivered_jobs"] = self.delivered
            summary["processing_completion_time"] = max(
                x.completion_time for x in result.schedule
            )
            write_json(
                self.run_dir / "execution_schedule.json",
                ExecutionScheduleFile(
                    schema=f"smartsom.execution-schedule/v{result.schedule_version}",
                    execution_schedule=result.execution_schedule,
                ),
            )
        if result.quality is not None:
            summary.update(
                passed_jobs=result.quality.passed_jobs,
                defective_jobs=result.quality.defective_jobs,
                total_jobs=result.quality.total_jobs,
                passing_rate=result.quality.passing_rate,
                quality=primitive(result.quality),
            )
        if solution is not None:
            summary.update(
                solver_status=solution.status,
                proven_optimal=solution.status == SolverStatus.OPTIMAL,
            )
        append_json(self.metrics_file, {"kind": "terminal", **summary})
        write_json(self.run_dir / "summary.json", summary)
        self.progress.write(f"completed makespan={result.makespan}\n")
        self.notify("completed", force=True)

    def finalize_manifest(self, status: str) -> None:
        self.manifest.update(status=status, artifacts=artifact_digests(self.run_dir))
        write_json(self.run_dir / "manifest.json", self.manifest)

    def fail(
        self, stage: str, exc: BaseException, records: Sequence[TraceRecord]
    ) -> None:
        # Actual records remain authoritative if a writer failed while draining.
        self.completed = sum(isinstance(record, CompletionRecord) for record in records)
        self.last_time = records[-1].simulation_time if records else 0
        failure = {
            "schema": "smartsom.failure/v1",
            "stage": stage,
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "simulation_time": self.last_time,
        }
        if hasattr(exc, "action"):
            failure["action"] = primitive(exc.action)
        try:
            write_json(self.run_dir / "failure.json", failure)
            write_json(
                self.run_dir / "summary.json",
                {
                    "schema": "smartsom.summary/v1",
                    "status": "failed",
                    "end_reason": "execution_failed",
                    "simulation_time": self.last_time,
                    "completed_operations": self.completed,
                    "makespan": None,
                },
            )
            with (self.run_dir / "progress.log").open(
                "a", encoding="utf-8"
            ) as progress:
                progress.write(f"failed stage={stage}: {exc}\n")
            self.finalize_manifest("failed")
        except OSError as evidence_error:
            exc.add_note(f"Could not finish failure evidence: {evidence_error}")
