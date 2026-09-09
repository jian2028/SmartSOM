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
from smartsom.learning.checkpoint import (
    CheckpointManifest,
    ResourceCheckpointManifest,
    file_hash,
)


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


def _stream_contract(run_dir, rows, resolved):
    path = run_dir / "training_streams.json"
    if not path.exists():
        controls_path = run_dir / "training_controls.json"
        if (
            controls_path.exists()
            and json.loads(controls_path.read_text()).get("num_envs", 1) != 1
        ):
            raise ValueError("training stream contract is missing")
        if [r["episode"] for r in rows] != list(range(len(rows))):
            raise ValueError("training episode ledger coverage/order mismatch")
        return 1
    streams = json.loads(path.read_text())
    controls = json.loads((run_dir / "training_controls.json").read_text())
    count = streams.get("num_envs")
    if (
        type(count) is not int
        or count < 2
        or streams
        != {
            "schema": "smartsom.training-streams/v1",
            "num_envs": count,
            "sampling_processes": controls["sampling_processes"],
            "episode_index": "local_episode * num_envs + stream_id",
            "collection_order": "vector_tick_then_stream_id",
            "steps_per_update": resolved.algorithm.algorithm.parameters.n_steps,
        }
        or controls["num_envs"] != count
    ):
        raise ValueError("training stream contract mismatch")
    indices = [0] * count
    partial = set()
    for row in rows:
        stream, local = row.get("stream_id"), row.get("local_episode")
        if (
            type(stream) is not int
            or not 0 <= stream < count
            or type(local) is not int
            or local != indices[stream]
            or row["episode"] != local * count + stream
            or stream in partial
        ):
            raise ValueError("training stream episode coverage/order mismatch")
        indices[stream] += 1
        if row["end_reason"] == "training_budget_stop":
            partial.add(stream)
    quotas = [
        sum(len(row["steps"]) for row in rows if row["stream_id"] == stream)
        for stream in range(count)
    ]
    if len(set(quotas)) != 1:
        raise ValueError("training streams have unequal sampling quotas")
    return count


def _audit_update_state(checkpoint_dir, checkpoint, decisions):
    directory = checkpoint_dir.parent
    if not (directory / "state_summary.json").exists():
        return {}
    from smartsom.experiments.training_lifecycle import inspect_resume_checkpoint
    from smartsom.learning.state_audit import checkpoint_state_summary

    manifest = inspect_resume_checkpoint(directory)
    summary = checkpoint_state_summary(checkpoint.provider, directory / "training")
    recorded = json.loads((directory / "state_summary.json").read_text())
    if (
        summary != recorded
        or summary["environment_steps"] != decisions
        or summary["learner_updates"] != checkpoint.learner_updates
        or manifest["environment_steps"] != decisions
    ):
        raise ValueError("serialized training state counters/optimizer disagree")
    return {"ppo_updates": manifest["ppo_updates"], "training_state": summary}


def audit_training(run_dir: Path) -> dict:
    """No learning, scientific resampling changes or file writes during the audit."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    lifecycle = (run_dir / "training_controls.json").is_file()
    allowed = (
        {"completed", "early_stopped", "pruned", "interrupted"}
        if lifecycle
        else {"completed"}
    )
    if (
        manifest["status"] not in allowed
        or artifact_digests(run_dir) != manifest["artifacts"]
    ):
        raise ValueError("training attempt is incomplete or evidence digests disagree")
    checkpoint_dir = run_dir / "checkpoint"
    if "checkpoint_dir" in manifest:
        if manifest["checkpoint_dir"] is None:
            raise ValueError("training has no saved checkpoint to audit model updates")
        from smartsom.experiments.packaging import locate_reference

        checkpoint_dir = locate_reference(
            run_dir / "manifest.json", manifest["checkpoint_dir"]
        )
    if file_hash(checkpoint_dir / "checkpoint.json") != manifest["checkpoint_sha256"]:
        raise ValueError("training checkpoint digest mismatch")
    resource = manifest["provider"] == "rllib.resource_ppo"
    checkpoint = read_model(
        checkpoint_dir / "checkpoint.json",
        ResourceCheckpointManifest if resource else CheckpointManifest,
    )[0]
    for entry in checkpoint.files:
        member = (checkpoint_dir / entry.path).resolve()
        if (
            not member.is_relative_to(checkpoint_dir.resolve())
            or file_hash(member) != entry.sha256
        ):
            raise ValueError("training checkpoint member digest mismatch")
    if checkpoint.initial_weights_sha256 == checkpoint.final_weights_sha256:
        raise ValueError("no learner parameter update")
    if resource and (
        any(w.initial_sha256 == w.final_sha256 for w in checkpoint.role_weights)
        or digest({w.role: w.initial_sha256 for w in checkpoint.role_weights})
        != checkpoint.initial_weights_sha256
        or digest({w.role: w.final_sha256 for w in checkpoint.role_weights})
        != checkpoint.final_weights_sha256
    ):
        raise ValueError("resource role parameter update mismatch")
    resolved = load_training_snapshot(run_dir / "resolved_training.json")
    rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    num_envs = _stream_contract(run_dir, rows, resolved)
    decisions = 0
    for row in rows:
        realization = resolved.episode(row["episode"])
        if (
            row["input_sha256"] != realization.input_sha256
            or row["seeds"] != primitive(realization.seeds)
            or row["root_seed"] != realization.root_seed
        ):
            raise ValueError("episode materialization or seed mismatch")
        if resource:
            from smartsom.experiments.resource_audit import audit_resource_episode

            audit_resource_episode(resolved, realization.input, row)
            decisions += len(row["steps"])
            continue
        from smartsom.learning.gymnasium import SchedulingEnv

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
    budget = resolved.run.budget.environment_steps
    completed_budget = decisions == budget
    if (
        checkpoint.environment_steps != decisions
        or decisions > budget
        or manifest["status"] == "completed"
        and not completed_budget
        or lifecycle
        and decisions % resolved.algorithm.algorithm.parameters.n_steps
        or not lifecycle
        and not completed_budget
    ):
        raise ValueError("training ledger sampling budget mismatch")
    if resource and (
        checkpoint.agent_steps
        != sum(len(s["indices"]) for r in rows for s in r["steps"])
        or checkpoint.physical_actions
        != sum(len(s["actions"]) for r in rows for s in r["steps"])
    ):
        raise ValueError("resource ledger agent/physical count mismatch")
    state = _audit_update_state(checkpoint_dir, checkpoint, decisions)
    if state and state["ppo_updates"] != (
        decisions // resolved.algorithm.algorithm.parameters.n_steps
    ):
        raise ValueError("checkpoint update boundary disagrees with sampling quota")
    return {
        "status": "passed",
        "training_status": manifest["status"],
        "budget_completed": completed_budget,
        "planned_environment_steps": budget,
        "episodes": len(rows),
        "environment_steps": decisions,
        "num_envs": num_envs,
        **state,
        "provider": checkpoint.provider,
        "framework_seed": checkpoint.framework_seed,
        "learner_updates": checkpoint.learner_updates,
        "initial_weights_sha256": checkpoint.initial_weights_sha256,
        "final_weights_sha256": checkpoint.final_weights_sha256,
        "completed_episodes": sum(r["end_reason"] == "completed" for r in rows),
        "failed_episodes": sum(
            r["end_reason"] not in ("completed", "training_budget_stop") for r in rows
        ),
        "checkpoint": str(checkpoint_dir),
        "source": manifest["source"],
        **(
            {
                "agent_steps": checkpoint.agent_steps,
                "physical_actions": checkpoint.physical_actions,
                "role_weights": primitive(checkpoint.role_weights),
                "team_return_definition": "one shared return; never sum over resources",
            }
            if resource
            else {}
        ),
    }
