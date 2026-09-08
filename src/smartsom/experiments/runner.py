"""Single-run orchestration; all simulation transitions remain in step()."""

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.algorithms import ScriptedPolicy
from smartsom.algorithms.solver import SolveRequest
from smartsom.config import ResolvedRun
from smartsom.engine import SimulationResult, Simulator
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.experiments.evidence import RunEvidence
from smartsom.experiments.providers import build_provider


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    simulation_result: SimulationResult


class RunFailedError(RuntimeError):
    def __init__(self, run_dir: Path, cause: BaseException):
        self.run_dir = run_dir
        self.cause = cause
        super().__init__(f"run failed in {run_dir}: {cause}")


def run_one(resolved_run: ResolvedRun, *, on_progress=None) -> RunResult:
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
    solution = None
    evidence = RunEvidence(run_dir, resolved, on_progress)
    try:
        with ExitStack() as stack:
            evidence.initialize(stack)
            provider = build_provider(resolved.algorithm)
            evidence.record_provider(provider)
            stage = "simulation"
            evidence.start_execution(stack)
            if resolved.algorithm.algorithm.interface_kind == "offline_solver":
                stage = "solving"
                request = SolveRequest(
                    resolved.factory,
                    resolved.workload,
                    resolved.run.objective,
                    resolved.run.budget.solver_time_limit_seconds,
                    next(
                        seed.value for seed in resolved.seeds if seed.domain == "solver"
                    ),
                )
                evidence.record_solver_request(request)
                solution = provider.solve(request)
                evidence.record_solver_result(solution)
                stage = "schedule_validation"
                solution.require_incumbent()
                policy = ScheduleReplayPolicy(
                    resolved.factory, resolved.workload, solution.schedule
                )
            else:
                policy = provider
            stage = "simulation"
            simulator = Simulator(
                resolved.factory,
                resolved.workload,
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
            evidence.drain(simulator.trace_since(evidence.trace_cursor))
            context = simulator.current_decision
            while context is not None:
                try:
                    evidence.observe(context)
                    outcome = simulator.step(policy.select_action(context))
                finally:
                    evidence.drain(simulator.trace_since(evidence.trace_cursor))
                evidence.record_progress()
                if isinstance(outcome, SimulationResult):
                    result = outcome
                    break
                context = outcome
            if isinstance(policy, ScriptedPolicy):
                policy.ensure_exhausted()
            if isinstance(policy, ScheduleReplayPolicy):
                stage = "schedule_verification"
                policy.verify_result(result)
            stage = "finalization"
            evidence.complete(result, solution)
        evidence.finalize_manifest("completed")
        return RunResult(run_dir, result)
    except (Exception, KeyboardInterrupt) as exc:
        evidence.fail(
            stage, exc, simulator.trace_since(0) if simulator is not None else ()
        )
        raise RunFailedError(run_dir, exc) from exc
