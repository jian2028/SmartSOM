"""Fixed validation worlds and conservative checkpoint selection, outside PPO."""

from dataclasses import replace
from statistics import mean

from smartsom.config.codec import digest
from smartsom.config.materialization import (
    materialize_arrivals,
    materialize_machine_events,
    materialize_processing_times,
    materialize_quality,
)
from smartsom.config.models import (
    GeneratedArrivals,
    GeneratedMachineEvents,
    GeneratedProcessingTimes,
    GeneratedQuality,
    MachineEventFile,
    ProcessingTimeFile,
    QualityFile,
)
from smartsom.config.seeds import derive_seeds
from smartsom.config.study import study_roots
from smartsom.config.training import episode_input


def validation_inputs(resolved, controls):
    """Study-world seeds materialize disturbances over the frozen training workload."""
    base, scenario = resolved.base, resolved.base.scenario
    inputs = []
    for replication in range(controls.replications):
        world, _ = study_roots(controls.seed, controls.case_id, replication, "")
        seeds = {
            s.domain: s.value
            for s in derive_seeds(
                world, generated=False, quality=base.quality is not None
            )
        }
        arrivals = materialize_arrivals(
            base.workload,
            scenario.arrivals.profile
            if isinstance(scenario.arrivals, GeneratedArrivals)
            else base.arrivals,
            seeds["demand"],
            base.arrival_provenance,
        )
        processing = materialize_processing_times(
            base.workload,
            scenario.processing_time.profile
            if isinstance(scenario.processing_time, GeneratedProcessingTimes)
            else ProcessingTimeFile(
                schema="smartsom.processing-times/v1",
                processing_times=base.processing_times,
                provenance=base.processing_provenance,
            )
            if base.processing_times
            else None,
            seeds["processing_time"],
        )
        machines = materialize_machine_events(
            base.factory,
            scenario.machine_events.profile
            if isinstance(scenario.machine_events, GeneratedMachineEvents)
            else MachineEventFile(
                schema="smartsom.machine-events/v1",
                machine_events=base.machine_events,
                provenance=base.machine_event_provenance,
            )
            if base.machine_events is not None
            else None,
            seeds["machine_events"],
        )
        quality = materialize_quality(
            base.factory,
            base.workload,
            processing.plan,
            scenario.quality
            if isinstance(scenario.quality, GeneratedQuality)
            else QualityFile(
                schema="smartsom.quality-draws/v1",
                draws=base.quality.draws,
                provenance=base.quality_provenance,
            )
            if base.quality
            else None,
            seeds.get("quality"),
        )
        episode = replace(
            episode_input(base),
            arrivals=arrivals.plan,
            processing_times=processing.plan,
            machine_events=machines.plan,
            quality=quality.plan,
        )
        inputs.append(
            {
                "input_id": digest([world, replication, episode]),
                "world_seed": world,
                "replication": replication,
                "episode": episode,
            }
        )
    return tuple(inputs)


def select_best(candidate, incumbent, controls):
    """Return (replace, reason); never compare survivor means from different cases."""
    successful = candidate["successful_inputs"]
    if not successful:
        return False, "no_completed_episodes"
    strict = controls.best_mode == "all_complete" or (
        controls.best_mode == "custom" and controls.failure_policy == "ineligible"
    )
    if strict and candidate["completed"] != candidate["episodes"]:
        return False, "requires_all_complete"
    if incumbent is None:
        return True, "first_eligible"
    if controls.best_mode == "completion_first":
        if candidate["completed"] != incumbent["completed"]:
            return candidate["completed"] > incumbent["completed"], "completion_count"
    if successful != incumbent["successful_inputs"]:
        return False, "different_successful_inputs"
    metric = controls.metric if controls.best_mode == "custom" else "makespan"
    direction = controls.direction if controls.best_mode == "custom" else "min"
    current, previous = candidate["metrics"][metric], incumbent["metrics"][metric]
    if current is None or previous is None:
        raise ValueError(f"validation metric is unavailable: {metric}")
    improvement = previous - current if direction == "min" else current - previous
    return (
        improvement > controls.min_delta,
        "metric" if improvement > controls.min_delta else "tie_or_no_improvement",
    )


def evaluate_inputs(resolved, controls, inputs, predict):
    from smartsom.learning.episode import EpisodeStartFailure

    resource = resolved.algorithm.algorithm.provider == "rllib.resource_ppo"
    if resource:
        from smartsom.learning.pettingzoo import SmartSOMParallelEnv

        env_type = SmartSOMParallelEnv
    else:
        from smartsom.learning.gymnasium import SchedulingEnv

        env_type = SchedulingEnv
    spec = resolved.algorithm.algorithm.projection
    rows = []
    for item in inputs:
        env = env_type(
            item["episode"],
            spec,
            limits=resolved.run.budget.limits(),
            **({} if resource else {"strict_actions": True}),
        )
        try:
            try:
                env.reset()
            except EpisodeStartFailure:
                pass
            else:
                while not env.finished:
                    env.step(predict(env.projected))
            row = {
                "input_id": item["input_id"],
                "world_seed": item["world_seed"],
                "replication": item["replication"],
                "reason": env.reason,
                "return": env.total_reward,
                "makespan": env.result.makespan if env.reason == "completed" else None,
                "passing_rate": env.result.quality.passing_rate
                if env.reason == "completed" and env.result.quality is not None
                else None,
                "steps": len(env.steps),
                "trace_sha256": digest(env.simulator.trace) if env.simulator else None,
            }
            if controls.full_replay and env.simulator is not None:
                if not env.steps:
                    replayed = env_type(
                        item["episode"], spec, limits=resolved.run.budget.limits()
                    )
                    try:
                        try:
                            replayed.reset()
                        except EpisodeStartFailure:
                            pass
                        if (
                            replayed.reason != env.reason
                            or replayed.total_reward != env.total_reward
                            or digest(replayed.simulator.trace) != row["trace_sha256"]
                        ):
                            raise ValueError(
                                "validation initial outcome replay differs"
                            )
                    finally:
                        replayed.close()
                elif resource:
                    from smartsom.config.codec import primitive
                    from smartsom.learning.joint_evidence import step_record
                    from smartsom.learning.joint_replay import replay_joint

                    replayed = replay_joint(
                        item["episode"],
                        spec,
                        [
                            primitive(step_record(i, s, full=False))
                            for i, s in enumerate(env.steps)
                        ],
                        limits=resolved.run.budget.limits(),
                    )
                    if digest(replayed.trace) != row["trace_sha256"]:
                        raise ValueError("validation joint replay differs")
                else:
                    replayed = SchedulingEnv(
                        item["episode"],
                        spec,
                        limits=resolved.run.budget.limits(),
                        strict_actions=True,
                    )
                    replayed.reset()
                    for step in env.steps:
                        replayed.step(step.action_index)
                    if (
                        replayed.steps != env.steps
                        or digest(replayed.simulator.trace) != row["trace_sha256"]
                    ):
                        raise ValueError("validation replay differs")
                    replayed.close()
                row["replay"] = "passed"
            rows.append(row)
        finally:
            env.close()
    successes = [r for r in rows if r["reason"] == "completed"]
    return {
        "episodes": len(rows),
        "completed": len(successes),
        "results": rows,
        "successful_inputs": sorted(r["input_id"] for r in successes),
        "metrics": {
            name: mean(r[name] for r in successes)
            if successes and all(r[name] is not None for r in successes)
            else None
            for name in ("makespan", "return", "passing_rate")
        },
    }


def predictor(provider, directory, hashes, *, deterministic, seed=None):
    """Inference-only loading with a private sampling RNG."""
    if provider == "sb3.maskable_ppo":
        from smartsom.learning.sb3 import load_predictor
    elif provider == "rllib.ppo":
        from smartsom.learning.rllib import load_predictor
    else:
        from smartsom.config.codec import read_model
        from smartsom.learning.checkpoint import ResourceCheckpointManifest
        from smartsom.learning.rllib_resource import load_predictor

        manifest = read_model(
            directory / "checkpoint.json", ResourceCheckpointManifest
        )[0]
        return load_predictor(
            directory, manifest, deterministic=deterministic, seed=seed
        )
    return load_predictor(directory, deterministic=deterministic, seed=seed)
