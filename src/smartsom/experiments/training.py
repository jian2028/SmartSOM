"""Training lifecycle and compact local evidence; simulation remains in Gym.step."""

import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import digest, primitive
from smartsom.config.training import ResolvedTrainingRun
from smartsom.experiments.evidence import (
    append_json,
    artifact_digests,
    source_identity,
    write_json,
)
from smartsom.learning.checkpoint import (
    CheckpointFile,
    CheckpointManifest,
    ResourceCheckpointManifest,
    file_hash,
    require_backend,
    structural_identity,
)
from smartsom.learning.joint_evidence import step_record


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    checkpoint_dir: Path | None
    environment_steps: int
    learner_updates: int
    status: str = "completed"
    last_checkpoint: Path | None = None
    best_checkpoint: Path | None = None


class TrainingFailedError(RuntimeError):
    def __init__(self, run_dir: Path, cause: BaseException):
        self.run_dir, self.cause = run_dir, cause
        super().__init__(f"training failed in {run_dir}: {cause}")


class TrainingEvidence:
    def __init__(self, run_dir, resolved, stack, on_progress):
        self.run_dir, self.resolved = run_dir, resolved
        self.on_progress = on_progress
        self.sampled_steps = self.updates = self.completed = self.failed = 0
        self.last_progress = 0.0
        self.active_env = None
        self.recorded = set()
        self.resource = resolved.algorithm.algorithm.provider == "rllib.resource_ppo"
        self.agent_steps = self.physical_actions = self.conflicts = 0
        self.role_weights = ()
        self.ledger = stack.enter_context(
            (run_dir / "episodes.jsonl").open("x", encoding="utf-8")
        )
        self.log = stack.enter_context(
            (run_dir / "progress.log").open("x", encoding="utf-8")
        )
        self.metrics = stack.enter_context(
            (run_dir / "learner_metrics.jsonl").open("x", encoding="utf-8")
        )
        self.debug = None
        if resolved.run.recording.debug:
            import logging
            from logging.handlers import RotatingFileHandler

            self.debug = logging.Logger(f"training.{run_dir.name}", level=logging.DEBUG)
            handler = RotatingFileHandler(
                run_dir / "debug.log", maxBytes=10 * 1024 * 1024, backupCount=2
            )
            self.debug.addHandler(handler)
            stack.callback(handler.close)

    def progress(self, stage, *, force=False):
        if not force and time.monotonic() - self.last_progress < 0.5:
            return
        self.last_progress = time.monotonic()
        row = {
            "stage": stage,
            "sampled_steps": self.sampled_steps,
            "learner_updates": self.updates,
            "completed_episodes": self.completed,
            "failed_episodes": self.failed,
        }
        if self.resource:
            row.update(
                agent_steps=self.agent_steps,
                physical_actions=self.physical_actions,
                conflicts=self.conflicts,
            )
        append_json(self.log, row)
        if self.debug:
            self.debug.debug(str(row))
        if self.on_progress:
            self.on_progress(row)

    def episode(self, env, *, partial_reason=None):
        key = env.episode_index
        if key in self.recorded:
            return
        realization = self.resolved.episode(key)
        reason = partial_reason or env.reason
        actions = (
            [a for s in env.steps for a in s.actions]
            if self.resource
            else [s.action for s in env.steps]
        )
        record = {
            "episode": key,
            **(
                {"stream_id": env.stream_id, "local_episode": env.local_episode}
                if getattr(self, "num_envs", 1) > 1
                else {}
            ),
            "seed_version": self.resolved.episode_seed_version,
            "root_seed": realization.root_seed,
            "seeds": realization.seeds,
            "input_sha256": realization.input_sha256,
            "bindings": env.projection.bindings,
            "steps": [
                step_record(
                    i, s, full=self.resolved.run.recording.observations == "full"
                )
                for i, s in enumerate(env.steps)
            ]
            if self.resource
            else [
                {
                    "index": s.action_index,
                    "action": s.action,
                    "reward": s.reward,
                    "simulation_time": s.simulation_time,
                    "reason": s.reason,
                    "observations_sha256": digest(s.projection.observations),
                    "mask_sha256": digest(s.projection.action_mask),
                    **(
                        {"observations": s.projection.observations}
                        if self.resolved.run.recording.observations == "full"
                        else {}
                    ),
                }
                for s in env.steps
            ],
            "actions_sha256": digest(actions),
            "trace_sha256": digest(env.simulator.trace if env.simulator else ()),
            "end_reason": reason,
            "return": env.total_reward,
            "makespan": env.result.makespan if reason == "completed" else None,
            "quality": env.result.quality if reason == "completed" else None,
        }
        if digest(env.input) != realization.input_sha256:
            raise ValueError("episode input differs from scientific materialization")
        append_json(self.ledger, record)
        self.recorded.add(key)
        if reason == "completed":
            self.completed += 1
        elif reason != "training_budget_stop":
            self.failed += 1
            write_json(
                self.run_dir / f"episode_{key:06d}_failure.json",
                {
                    "input": realization.input,
                    "record": record,
                    "trace": env.simulator.trace if env.simulator else (),
                },
            )
        self.progress("sampling", force=True)

    def learner(self, updates, metrics):
        numeric = {}

        def visit(value, path=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}/{key}" if path else str(key))
            elif isinstance(value, Real) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite learner metric: {path}")
                numeric[path] = float(value)

        visit(metrics)
        if not any("loss" in key for key in numeric):
            raise ValueError("learner did not report numeric losses")
        self.updates = updates
        append_json(
            self.metrics,
            {
                "update": updates,
                "sampled_steps": self.sampled_steps,
                "metrics": numeric,
            },
        )
        self.progress("learning", force=True)
        return numeric


def train_one(
    resolved_training_run: ResolvedTrainingRun, *, on_progress=None, controls=None
) -> TrainingResult:
    if not isinstance(resolved_training_run, ResolvedTrainingRun):
        raise TypeError("train_one accepts only ResolvedTrainingRun")
    resolved = resolved_training_run
    spec = resolved.algorithm.algorithm
    dependencies = require_backend(spec.provider)
    lifecycle = None
    if controls is not None:
        from smartsom.experiments.training_lifecycle import TrainingLifecycle

        lifecycle = TrainingLifecycle(resolved, controls)
        lifecycle.resume_dependencies = dependencies
    inputs = resolved.episode(0).input
    # Complete materialization and capacity validation before allocating evidence.
    resource = spec.provider == "rllib.resource_ppo"
    if resource:
        from smartsom.learning.pettingzoo import SmartSOMParallelEnv

        env = SmartSOMParallelEnv(
            inputs,
            spec.projection,
            limits=resolved.run.budget.limits(),
            episode_source=lambda index: resolved.episode(index).input,
        )
    else:
        from smartsom.learning.gymnasium import SchedulingEnv

        env = SchedulingEnv(
            inputs,
            spec.projection,
            limits=resolved.run.budget.limits(),
            observation_kind="masked" if spec.provider == "rllib.ppo" else "plain",
            episode_source=lambda index: resolved.episode(index).input,
            strict_actions=True,
        )
    root = Path(resolved.run.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
    )
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint"
    export_final = lifecycle is None
    manifest = {
        "schema": "smartsom.training-manifest/v1",
        "status": "running",
        "source": source_identity(),
        "dependencies": dependencies,
        "provider": spec.provider,
        "framework_seed": resolved.framework_seed,
        "episode_seed_version": resolved.episode_seed_version,
        "base_workload_sha256": resolved.base.workload_sha256,
        "device": controls.device if controls else "cpu",
        "numerical_threads": controls.numerical_threads if controls else 1,
        "budget": primitive(resolved.run.budget),
    }
    evidence = None
    try:
        write_json(run_dir / "manifest.json", manifest)
        write_json(
            run_dir / "resolved_training.json",
            {"schema": "smartsom.resolved-training/v1", "resolved": resolved},
        )
        if export_final:
            checkpoint.mkdir()
        with ExitStack() as stack:
            evidence = TrainingEvidence(run_dir, resolved, stack, on_progress)
            if lifecycle:
                lifecycle.bind(evidence, stack)
            env.on_episode = evidence.episode
            evidence.active_env = env
            if resource:
                from smartsom.learning.rllib_resource import train
            elif spec.provider == "rllib.ppo":
                from smartsom.learning.rllib import train
            else:
                from smartsom.learning.sb3 import train
            try:
                before, after, steps, updates = train(
                    resolved,
                    env,
                    evidence,
                    checkpoint,
                    **({"lifecycle": lifecycle} if lifecycle else {}),
                )
                active = evidence.active_env
                for current in getattr(evidence, "active_envs", [active]):
                    if (
                        current is not None
                        and current.simulator is not None
                        and not current.finished
                    ):
                        evidence.episode(current, partial_reason="training_budget_stop")
                if (
                    before == after
                    or updates <= 0
                    or (
                        steps != resolved.run.budget.environment_steps
                        and not (lifecycle and lifecycle.stopped)
                    )
                ):
                    raise ValueError(
                        "training failed its parameter-update or budget gate"
                    )
                if export_final:
                    files = tuple(
                        CheckpointFile(
                            path=str(p.relative_to(checkpoint)), sha256=file_hash(p)
                        )
                        for p in sorted(checkpoint.rglob("*"))
                        if p.is_file()
                    )
                    metadata = dict(
                        schema="smartsom.resource-checkpoint/v1"
                        if resource
                        else "smartsom.checkpoint/v1",
                        provider=spec.provider,
                        projection=spec.projection,
                        parameters=spec.parameters,
                        structure_sha256=structural_identity(
                            resolved.base, spec.projection
                        ),
                        files=files,
                        dependencies=tuple(sorted(dependencies.items())),
                        initial_weights_sha256=before,
                        final_weights_sha256=after,
                        environment_steps=steps,
                        learner_updates=updates,
                        framework_seed=resolved.framework_seed,
                    )
                    if resource:
                        metadata.update(
                            role_mapping=tuple(
                                (a, active.policy_for_agent(a))
                                for a in active.possible_agents
                            ),
                            role_weights=evidence.role_weights,
                            agent_steps=evidence.agent_steps,
                            physical_actions=evidence.physical_actions,
                        )
                    checkpoint_manifest = (
                        ResourceCheckpointManifest if resource else CheckpointManifest
                    )(**metadata)
                    write_json(checkpoint / "checkpoint.json", checkpoint_manifest)
                    write_json(
                        run_dir / "checkpoint_algorithm.json",
                        resolved.algorithm.model_copy(
                            update={
                                "algorithm": spec.model_copy(
                                    update={
                                        "checkpoint": "checkpoint",
                                        "checkpoint_sha256": file_hash(
                                            checkpoint / "checkpoint.json"
                                        ),
                                    }
                                )
                            }
                        ),
                    )
                write_json(
                    run_dir / "summary.json",
                    {
                        "status": lifecycle.status if lifecycle else "completed",
                        "environment_steps": steps,
                        "learner_updates": updates,
                        **({"ppo_updates": lifecycle.ppo_updates} if lifecycle else {}),
                        "completed_episodes": evidence.completed,
                        "failed_episodes": evidence.failed,
                        "initial_weights_sha256": before,
                        "final_weights_sha256": after,
                        **(
                            {
                                "agent_steps": evidence.agent_steps,
                                "physical_actions": evidence.physical_actions,
                                "conflicts": evidence.conflicts,
                                "role_weights": evidence.role_weights,
                            }
                            if resource
                            else {}
                        ),
                    },
                )
                evidence.progress(
                    lifecycle.status if lifecycle else "completed", force=True
                )
            except BaseException:
                for active in getattr(evidence, "active_envs", [evidence.active_env]):
                    if active is None or active.simulator is None:
                        continue
                    try:
                        evidence.episode(active, partial_reason="training_error")
                    except Exception:
                        pass  # Preserve the original failure if recording also fails.
                raise
        if not export_final:
            selected = lifecycle.last_checkpoint or lifecycle.best_checkpoint
            checkpoint = selected / "inference" if selected else None
            manifest.update(
                checkpoint_dir=str(checkpoint) if checkpoint else None,
                checkpoint_files=[
                    primitive(
                        CheckpointFile(
                            path=str(path.relative_to(checkpoint)),
                            sha256=file_hash(path),
                        )
                    )
                    for path in sorted(checkpoint.rglob("*"))
                    if path.is_file() and path.name != "checkpoint.json"
                ]
                if checkpoint
                else [],
            )
        manifest.update(
            status=lifecycle.status if lifecycle else "completed",
            artifacts=artifact_digests(run_dir),
            checkpoint_sha256=file_hash(checkpoint / "checkpoint.json")
            if checkpoint
            else None,
        )
        write_json(run_dir / "manifest.json", manifest)
        return TrainingResult(
            run_dir,
            checkpoint,
            steps,
            updates,
            lifecycle.status if lifecycle else "completed",
            lifecycle.last_checkpoint if lifecycle else None,
            lifecycle.best_checkpoint if lifecycle else None,
        )
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "status": "failed",
            "exception": type(exc).__name__,
            "reason": str(exc),
        }
        # Evidence errors must not replace the original training failure.
        try:
            write_json(run_dir / "failure.json", failure)
            write_json(run_dir / "summary.json", failure)
            manifest.update(status="failed", artifacts=artifact_digests(run_dir))
            write_json(run_dir / "manifest.json", manifest)
        except OSError:
            pass
        raise TrainingFailedError(run_dir, exc) from exc
