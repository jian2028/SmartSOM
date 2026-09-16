"""One production execution loop for rules, learned policies, recording and audit."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.algorithms.production import (
    ScriptedProductionPolicy,
)
from smartsom.config.production import frozen_inputs
from smartsom.engine.production import ProductionSimulator
from smartsom.trace.production import Recorder, observation_record


@dataclass(frozen=True)
class ProductionResult:
    makespan: int | None
    status: str
    total_reward: float
    qualified_demands: int
    final_state: dict


class TerminalDisplay:
    def __init__(self, scenario, enabled, debug=False):
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TimeElapsedColumn,
        )

        self.enabled, self.scenario = enabled, scenario
        self.debug = debug
        self.console = Console(stderr=True)
        self.progress = None
        if enabled and self.console.is_terminal:
            total = (
                len(scenario.demands)
                if scenario.mode == "static"
                else scenario.tick_limit
            )
            self.progress = Progress(
                "{task.description}",
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            )
            self.task = self.progress.add_task(scenario.mode, total=total)
            self.progress.start()

    def update(self, row):
        if not self.enabled:
            return
        if self.debug:
            from smartsom.trace.production import canonical

            self.console.print(
                canonical(row), markup=False, highlight=False, soft_wrap=True
            )
        for event in row["events"]:
            if event["kind"] != "move":
                self.console.print(
                    f"tick {event['tick']:>6}  {event['kind']}  "
                    + " ".join(
                        f"{k}={v}"
                        for k, v in event.items()
                        if k not in ("tick", "kind", "defect", "actual_ticks")
                    ),
                    markup=False,
                    highlight=False,
                )
        if self.progress:
            completed = (
                len(row["state"]["completed"])
                if self.scenario.mode == "static"
                else row["tick"]
            )
            self.progress.update(
                self.task,
                completed=completed,
                description=f"{self.scenario.mode} · tick {row['tick']}",
            )

    def close(self):
        if self.progress:
            self.progress.stop()


def execute(
    scenario,
    algorithm,
    *,
    output_root="runs",
    name="production",
    record=True,
    verbose=True,
    policy=None,
    full_replay=False,
    on_progress=None,
    context=None,
    experiment=None,
    debug=False,
    policy_seed=None,
    limits=None,
    checkpoint_identity=None,
    deterministic=True,
    input_metadata=None,
    observations=None,
):
    from smartsom.experiments.evidence import source_identity

    if any(type(value) is not bool for value in (record, verbose, debug, full_replay)):
        raise ValueError("record, verbose, debug and full_replay must be booleans")
    if observations not in (None, "hash", "full"):
        raise ValueError("observations must be hash or full")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("run name must be a single alphanumeric slug")
    sim = ProductionSimulator(scenario)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root).resolve() / f"{stamp}-{name}-{uuid4().hex[:8]}"
    recorder = Recorder(
        directory,
        frozen_inputs(scenario, algorithm),
        sim.snapshot(),
        record=record,
        source=None,
    )
    if input_metadata is not None:
        recorder.manifest["input_metadata"] = input_metadata
    recorder.manifest["observations"] = observations
    display = None
    owned_policy = None
    primary_error = None
    decisions = 0
    limited = False
    try:
        recorder.manifest["stage"] = "initialization"
        recorder.manifest["source"] = source_identity()
        if policy is None:
            from smartsom.experiments.providers import build_provider

            policy = build_provider(
                algorithm,
                scenario,
                seed=policy_seed,
                deterministic=deterministic,
                limits=limits,
                observations=observations,
            )
            if algorithm.provider.startswith(("sb3.", "rllib.")):
                owned_policy = policy
        if hasattr(policy, "next_tick"):
            sim = policy.sim
        learning_contract = getattr(policy, "learning_contract", None)
        if (
            algorithm.provider.startswith(("sb3.", "rllib."))
            and learning_contract is None
        ):
            raise ValueError(
                "learned execution requires its frozen input evidence contract"
            )
        if learning_contract:
            recorder.manifest["learning_contract"] = learning_contract
        if checkpoint_identity:
            recorder.manifest["checkpoint"] = checkpoint_identity
        elif owned_policy:
            from smartsom.learning.checkpoint import file_hash

            recorder.manifest["checkpoint"] = {
                "path": str(Path(algorithm.checkpoint).resolve()),
                "provider": algorithm.provider,
                "manifest_sha256": file_hash(
                    Path(algorithm.checkpoint) / "checkpoint.json"
                ),
            }
        recorder.manifest["algorithm_seed"] = (
            scenario.seed if policy_seed is None else policy_seed
        )
        if experiment is not None:
            from smartsom.trace.production import atomic_json

            recorder.manifest["experiment"] = experiment
            atomic_json(directory / "run.json", recorder.manifest)
        display = TerminalDisplay(scenario, verbose, debug)
        checker = None
        if full_replay:
            from smartsom.trace.production import ExecutionAudit

            checker = ExecutionAudit(
                scenario, sim.snapshot(), learning_contract, observations=observations
            )
        recorder.manifest["stage"] = "simulation"
        while not sim.done:
            if hasattr(policy, "next_tick"):
                row = policy.next_tick()
                if row is None:
                    break
            else:
                ranking_view = sim.decision()
                ranking_record = (
                    observation_record(ranking_view, observations)
                    if observations
                    else None
                )
                rankings = policy.rank(ranking_view)
                view = sim.decision(rankings)
                action_record = (
                    observation_record(view, observations) if observations else None
                )
                command = policy.act(view)
                row = sim.step(command)
                row["buffer_scores"] = getattr(policy, "scores", {})
                if observations:
                    row["rule_decision"] = {
                        "ranking": ranking_record,
                        "action": action_record,
                    }
                decisions += 1
            recorder.append(row)
            if isinstance(policy, ScriptedProductionPolicy) and row["rejections"]:
                raise ValueError(
                    f"scripted command rejected at tick {row['tick']}: {row['rejections']}"
                )
            if checker:
                checker.append(row)
            display.update(row)
            if on_progress:
                on_progress(
                    {
                        "stage": "simulation",
                        "tick": row["tick"],
                        "status": row["status"],
                        "qualified_demands": len(sim.completed),
                        "run_dir": str(directory),
                    }
                )
            if limits and not sim.done and not hasattr(policy, "next_tick"):
                limited = (
                    decisions >= limits.max_decisions or sim.tick >= limits.max_ticks
                )
                if limited:
                    break
        if (
            sim.status == "completed"
            and isinstance(policy, ScriptedProductionPolicy)
            and policy.position != len(policy.commands)
        ):
            raise ValueError("script has extra commands after completion")
        limited = limited or (
            hasattr(policy, "env") and getattr(policy.env, "limit_hit", False)
        )
        status = sim.status if sim.done else "truncated" if limited else "interrupted"
        recorder.manifest["reason"] = "budget_exhausted" if limited else status
        recorder.manifest["context"] = context
        pending = policy.env.pending_decisions if learning_contract else []
        if learning_contract:
            recorder.manifest["pending_decisions"] = pending
        if checker:
            checker.finish_pending(pending)
        recorder.manifest["audit"] = (
            checker.result() if checker else {"status": "not_requested"}
        )
        recorder.finish(status)
        if verbose:
            display.console.print(
                f"{status}: tick={sim.tick}, passed={len(sim.completed)}, run_dir={directory}",
                markup=False,
            )
        return directory
    except BaseException as exc:
        primary_error = exc
        exc.run_dir = directory
        recorder.manifest["execution_state"] = sim.snapshot()
        if getattr(policy, "learning_contract", None):
            recorder.manifest["pending_decisions"] = policy.env.pending_decisions
        try:
            recorder.finish(
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                {"type": type(exc).__name__, "message": str(exc)},
            )
        except Exception as metadata_error:
            exc.add_note(f"failure metadata could not be saved: {metadata_error}")
        raise
    finally:
        for resource in (display, owned_policy.env if owned_policy else None):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"run cleanup also failed: {cleanup_error}")


def run(scenario, algorithm, **kwargs):
    return execute(scenario, algorithm, **kwargs)
