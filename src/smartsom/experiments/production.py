"""One production execution loop for rules, learned policies, recording and UI."""

import re
import threading
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.algorithms.production import (
    ScriptedProductionPolicy,
)
from smartsom.config.production import frozen_inputs
from smartsom.engine.production import ProductionSimulator
from smartsom.telemetry.runtime import CURRENT, backend_diagnostics, bind, operation
from smartsom.trace.production import Recorder, observation_record


@dataclass(frozen=True)
class ProductionResult:
    makespan: int | None
    status: str
    total_reward: float
    qualified_demands: int
    final_state: dict


class RunControls:
    def __init__(self):
        self.condition = threading.Condition()
        self.paused = False
        self.stopped = False
        self.steps = 0
        self.delay = 0.1
        self.latest = None
        self.finished = False
        self.error = None
        self.outcome = None

    def permission(self):
        with self.condition:
            while self.paused and not self.steps and not self.stopped:
                self.condition.wait()
            if self.steps:
                self.steps -= 1
            return not self.stopped

    def pause(self, value):
        with self.condition:
            self.paused = value
            self.condition.notify_all()

    def step(self):
        with self.condition:
            self.paused = True
            self.steps += 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def detach(self):
        with self.condition:
            self.delay = 0
            self.paused = False
            self.condition.notify_all()


class TerminalDisplay:
    def __init__(self, scenario, enabled, debug=False, *, directory=None):
        self.scenario = scenario
        self.session = CURRENT.get()
        self.task = str(directory)
        self.console = self.session.console
        self.total = (
            len(scenario.demands) if scenario.mode == "static" else scenario.tick_limit
        )
        self.unit = (
            "qualified deliveries" if scenario.mode == "static" else "physical ticks"
        )

    def update(self, row):
        self.session.update(
            self.task,
            {
                "stage": "simulation",
                "tick": row["tick"],
                "qualified_demands": len(row["state"]["completed"]),
            },
            total=self.total,
            unit=self.unit,
        )
        if self.session.options.debug:
            self.session.diagnostic(row)

    def close(self):
        pass


@operation("run")
def execute(
    scenario,
    algorithm,
    *,
    output_root="runs",
    name="production",
    record=True,
    verbose=True,
    controls=None,
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
    bind(directory, name)
    try:
        recorder.manifest["stage"] = "initialization"
        recorder.manifest["source"] = source_identity()
        if policy is None:
            from smartsom.experiments.providers import build_provider

            with backend_diagnostics():
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
        display = TerminalDisplay(scenario, verbose, debug, directory=directory)
        checker = None
        if full_replay:
            from smartsom.trace.production import ExecutionAudit

            checker = ExecutionAudit(
                scenario, sim.snapshot(), learning_contract, observations=observations
            )
        if controls:
            controls.context = context
            controls.latest = {"tick": 0, "state": sim.snapshot(), "events": []}
        recorder.manifest["stage"] = "simulation"
        while not sim.done:
            if controls and not controls.permission():
                break
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
            if controls:
                controls.latest = row
                if controls.delay:
                    # Wall-clock presentation pacing never enters core physics.
                    with controls.condition:
                        controls.condition.wait(timeout=controls.delay)
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
        if controls:
            controls.outcome = recorder.manifest["reason"]
        display.session.update(
            str(directory),
            {
                "stage": status,
                "status": status,
                "reason": (
                    "Decision budget exhausted"
                    if limited
                    and learning_contract
                    and policy.env.limits is not None
                    and policy.env.decisions >= policy.env.limits.max_decisions
                    else recorder.manifest["reason"]
                ),
                "display_values": (
                    {
                        "decisions": policy.env.decisions,
                        "decision_limit": policy.env.limits.max_decisions
                        if policy.env.limits
                        else None,
                    }
                    if learning_contract
                    else {}
                ),
            },
            final=True,
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
        if controls:
            controls.error = exc
        raise
    finally:
        if controls:
            controls.finished = True
        for resource in (display, owned_policy.env if owned_policy else None):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"run cleanup also failed: {cleanup_error}")


def run(scenario, algorithm, *, render_mode=None, **kwargs):
    if render_mode is None:
        return execute(scenario, algorithm, **kwargs)
    if render_mode != "human":
        raise ValueError("render_mode must be None or human")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("human rendering must be launched from the main thread")
    from smartsom.studio.playback import live_window

    controls = RunControls()
    controls.context = kwargs.get("context")
    result = []

    def worker():
        try:
            result.append(execute(scenario, algorithm, controls=controls, **kwargs))
        except BaseException as exc:
            controls.error = exc
            controls.finished = True

    copy = copy_context()
    thread = threading.Thread(target=lambda: copy.run(worker), name="smartsom-physics")
    # Construct the window before starting work so startup/import errors cannot
    # leave an uncontrolled simulation running in a background thread.
    live_window(scenario.factory, controls, thread)
    thread.join()
    if controls.error:
        raise controls.error
    return result[0]
