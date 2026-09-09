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
from smartsom.experiments.evidence import RunEvidence, append_json
from smartsom.experiments.providers import build_provider
from smartsom.learning.resource_policy import ResourceCheckpointPolicy


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    simulation_result: SimulationResult


class RunFailedError(RuntimeError):
    def __init__(self, run_dir: Path, cause: BaseException):
        self.run_dir = run_dir
        self.cause = cause
        super().__init__(f"run failed in {run_dir}: {cause}")


def run_one(
    resolved_run: ResolvedRun, *, on_progress=None, deterministic=True
) -> RunResult:
    if not isinstance(resolved_run, ResolvedRun):
        raise TypeError("run_one accepts only ResolvedRun")
    resolved = resolved_run
    from smartsom.learning.checkpoint import CheckpointPolicy, validate_checkpoint

    checkpoint_manifest = validate_checkpoint(resolved)
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
            if checkpoint_manifest:
                policy_type = (
                    ResourceCheckpointPolicy
                    if resolved.algorithm.algorithm.provider == "rllib.resource_ppo"
                    else CheckpointPolicy
                )
                provider = (
                    policy_type(resolved)
                    if deterministic
                    else policy_type(resolved, deterministic=False)
                )
            else:
                provider = build_provider(resolved.algorithm)
            evidence.record_provider(provider)
            if checkpoint_manifest:
                evidence.manifest["learning_checkpoint"] = {
                    "manifest_sha256": resolved.algorithm.algorithm.checkpoint_sha256,
                    "provider": checkpoint_manifest.provider,
                    "projection": checkpoint_manifest.projection,
                    "structure_sha256": checkpoint_manifest.structure_sha256,
                    "dependencies": checkpoint_manifest.dependencies,
                    "deterministic": deterministic,
                }
            stage = "simulation"
            evidence.start_execution(stack)
            if checkpoint_manifest and checkpoint_manifest.extensions is not None:
                extension_file = stack.enter_context(
                    (run_dir / "extension_decisions.jsonl").open("x", encoding="utf-8")
                )
                provider.on_extension = lambda row: append_json(extension_file, row)
                reward_file = stack.enter_context(
                    (run_dir / "extension_rewards.jsonl").open("x", encoding="utf-8")
                )
                provider.on_reward = lambda row: append_json(reward_file, row)
                evidence.manifest["learning_checkpoint"]["extensions"] = (
                    checkpoint_manifest.extensions
                )
            if isinstance(provider, ResourceCheckpointPolicy):
                joint_file = stack.enter_context(
                    (run_dir / "joint_decisions.jsonl").open("x", encoding="utf-8")
                )
                provider.on_round = lambda row: append_json(joint_file, row)
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
            try:
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
            except BaseException as exc:
                if checkpoint_manifest and checkpoint_manifest.extensions is not None:
                    records = simulator.trace_since(0) if simulator is not None else ()
                    try:
                        policy.fail(
                            exc,
                            tick=max((r.simulation_time for r in records), default=0),
                            trace_end=len(records),
                        )
                    except Exception:
                        pass
                raise
            while context is not None:
                try:
                    evidence.observe(context)
                    if isinstance(policy, ResourceCheckpointPolicy):
                        policy.observed_trace_end = evidence.trace_cursor
                    outcome = simulator.step(policy.select_action(context))
                    if isinstance(policy, CheckpointPolicy):
                        policy.check_outcome(outcome)
                    elif isinstance(policy, ResourceCheckpointPolicy):
                        policy.check_outcome(
                            outcome,
                            trace_end=evidence.trace_cursor
                            + len(simulator.trace_since(evidence.trace_cursor)),
                        )
                except BaseException as exc:
                    if isinstance(policy, ResourceCheckpointPolicy) or (
                        isinstance(policy, CheckpointPolicy)
                        and policy.extensions is not None
                    ):
                        records = simulator.trace_since(evidence.trace_cursor)
                        try:
                            policy.fail(
                                exc,
                                tick=max(
                                    (r.simulation_time for r in records),
                                    default=context.simulation_time,
                                ),
                                trace_end=evidence.trace_cursor + len(records),
                            )
                        except Exception:
                            pass  # A second evidence failure cannot replace the original.
                    raise
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
