"""Regenerate scientific episodes and verify the compact learning ledger."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from smartsom.config.codec import digest, primitive, read_model
from smartsom.config.models import StrictModel
from smartsom.config.snapshots import validate_resolved
from smartsom.config.training import (
    EPISODE_SEED_VERSION,
    ResolvedTrainingRun,
    framework_seed,
)
from smartsom.engine import replay_schedule
from smartsom.experiments.evidence import artifact_digests
from smartsom.learning.checkpoint import CheckpointManifest, file_hash


class TrainingSnapshot(StrictModel):
    schema_id: Literal["smartsom.resolved-training/v1"] = Field(alias="schema")
    resolved: ResolvedTrainingRun


def load_training_snapshot(path: Path) -> ResolvedTrainingRun:
    resolved = read_model(path, TrainingSnapshot)[0].resolved
    validate_resolved(resolved.base)
    if (
        resolved.episode_seed_version != EPISODE_SEED_VERSION
        or resolved.framework_seed
        != framework_seed(resolved.run.seed, resolved.algorithm.algorithm.provider)
    ):
        raise ValueError("training snapshot seed authority mismatch")
    if resolved.base.run.seed != resolved.run.seed:
        raise ValueError("training snapshot base seed mismatch")
    return resolved


def audit_training(run_dir: Path) -> dict:
    """No learning, scientific resampling changes or file writes during the audit."""
    from smartsom.learning.gymnasium import SchedulingEnv

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (
        manifest["status"] != "completed"
        or artifact_digests(run_dir) != manifest["artifacts"]
    ):
        raise ValueError("training attempt is incomplete or evidence digests disagree")
    if (
        file_hash(run_dir / "checkpoint/checkpoint.json")
        != manifest["checkpoint_sha256"]
    ):
        raise ValueError("training checkpoint digest mismatch")
    checkpoint = read_model(run_dir / "checkpoint/checkpoint.json", CheckpointManifest)[
        0
    ]
    for entry in checkpoint.files:
        member = (run_dir / "checkpoint" / entry.path).resolve()
        if (
            not member.is_relative_to((run_dir / "checkpoint").resolve())
            or file_hash(member) != entry.sha256
        ):
            raise ValueError("training checkpoint member digest mismatch")
    if checkpoint.initial_weights_sha256 == checkpoint.final_weights_sha256:
        raise ValueError("no learner parameter update")
    resolved = load_training_snapshot(run_dir / "resolved_training.json")
    rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    indices = [r["episode"] for r in rows]
    if indices != list(range(len(rows))):
        raise ValueError("training episode ledger coverage/order mismatch")
    decisions = 0
    for row in rows:
        realization = resolved.episode(row["episode"])
        if (
            row["input_sha256"] != realization.input_sha256
            or row["seeds"] != primitive(realization.seeds)
            or row["root_seed"] != realization.root_seed
        ):
            raise ValueError("episode materialization or seed mismatch")
        env = SchedulingEnv(
            realization.input,
            resolved.algorithm.algorithm.projection,
            limits=resolved.run.budget.limits(),
            strict_actions=True,
        )
        env.reset()
        for expected in row["steps"]:
            if (
                digest(env.projected.observations) != expected["observations_sha256"]
                or digest(env.projected.action_mask) != expected["mask_sha256"]
            ):
                raise ValueError("learning observation or mask replay mismatch")
            action = env.projected.decode(expected["index"])
            if primitive(action) != expected["action"]:
                raise ValueError("learning slot/semantic action mismatch")
            _, reward, _, _, info = env.step(expected["index"])
            if (
                reward != expected["reward"]
                or info["simulation_time"] != expected["simulation_time"]
                or info["end_reason"] != expected["reason"]
            ):
                raise ValueError("learning transition/reward replay mismatch")
        decisions += len(row["steps"])
        if (
            digest(env.simulator.trace) != row["trace_sha256"]
            or primitive(env.projection.bindings) != row["bindings"]
        ):
            raise ValueError("episode trace or binding replay mismatch")
        if (
            digest([s.action for s in env.steps]) != row["actions_sha256"]
            or env.total_reward != row["return"]
        ):
            raise ValueError("episode action/return digest mismatch")
        if row["end_reason"] == "training_budget_stop":
            if env.finished:
                raise ValueError("partial training episode was actually terminal")
        elif env.reason != row["end_reason"]:
            raise ValueError("episode termination reason mismatch")
        if env.reason == "completed":
            inp = realization.input
            schedule = (
                env.result.execution_schedule
                if inp.transport_enabled or inp.buffers_enabled
                else env.result.schedule
            )
            replayed = replay_schedule(
                inp.factory, inp.workload, schedule, **inp.options()
            )
            if (
                replayed.execution_schedule != env.result.execution_schedule
                or replayed.quality != env.result.quality
                or replayed.makespan != row["makespan"]
            ):
                raise ValueError("training schedule replay mismatch")
    if (
        decisions != resolved.run.budget.environment_steps
        or checkpoint.environment_steps != decisions
    ):
        raise ValueError("training ledger sampling budget mismatch")
    return {
        "status": "passed",
        "episodes": len(rows),
        "environment_steps": decisions,
        "provider": checkpoint.provider,
        "framework_seed": checkpoint.framework_seed,
        "learner_updates": checkpoint.learner_updates,
        "initial_weights_sha256": checkpoint.initial_weights_sha256,
        "final_weights_sha256": checkpoint.final_weights_sha256,
        "completed_episodes": sum(r["end_reason"] == "completed" for r in rows),
        "failed_episodes": sum(
            r["end_reason"] not in ("completed", "training_budget_stop") for r in rows
        ),
        "checkpoint": str(run_dir / "checkpoint"),
        "source": manifest["source"],
    }
