"""Bounded local RLlib training + whole-episode sampling; no campaign launch."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def digest_tree(root):
    return hashlib.sha256(
        b"".join(
            str(p.relative_to(root)).encode() + b"\0" + p.read_bytes()
            for p in sorted(root.rglob("*.py"))
        )
    ).hexdigest()


def install_phases(algorithm):
    from phase_clock import PhaseClock
    from ray.rllib.core.learner.learner import Learner
    from ray.rllib.core.learner.torch.torch_learner import TorchLearner
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.policy.sample_batch import SampleBatch

    from smartsom.algorithms.training import TrainingComponents
    from smartsom.engine.base_protocol import BaseProtocol
    from smartsom.engine.central_control import CentralControl
    from smartsom.engine.production import ProductionSimulator
    from smartsom.learning.factory_episode import FactoryEpisode
    from smartsom.learning.rllib_central_env import CentralFactoryEnv
    from smartsom.learning.rllib_env import FactoryMultiAgentEnv
    from smartsom.learning.rllib_observation import V2ObservationCodec
    from smartsom.traffic.rule_mover import SharedMover

    clock = PhaseClock()
    for owner, method, label in (
        (algorithm.env_runner, "sample", "sampling_framework"),
        (algorithm.learner_group, "update", "learner_update"),
        (SampleBatch, "shuffle", "learner_batch_shuffle"),
        (RLModule, "forward_train", "learner_network_forward"),
        (Learner, "compute_losses", "learner_loss"),
        (TorchLearner, "compute_gradients", "learner_backward"),
        (TorchLearner, "apply_gradients", "learner_optimizer"),
        (algorithm.env_runner.module, "forward_exploration", "policy_inference"),
        (algorithm.env_runner.module, "forward_inference", "policy_inference"),
        (FactoryEpisode, "act", "factory_action_and_reward"),
        (FactoryEpisode, "wait", "factory_action_and_reward"),
        (FactoryEpisode, "_next", "decision_protocol"),
        (BaseProtocol, "make_request", "request_features"),
        (CentralControl, "candidates", "central_candidates"),
        (BaseProtocol, "advance", "physical_services"),
        (ProductionSimulator, "step", "simulation_core"),
        (SharedMover, "moves", "traffic_mover"),
        (FactoryMultiAgentEnv, "_observations", "numeric_observation"),
        (TrainingComponents, "value_input", "critic_observation"),
        (V2ObservationCodec, "encode", "actor_observation"),
        (V2ObservationCodec, "encode_wait", "wait_observation"),
        (CentralFactoryEnv, "_observation", "numeric_observation"),
        (FactoryMultiAgentEnv, "reset", "episode_reset"),
        (CentralFactoryEnv, "reset", "episode_reset"),
    ):
        clock.wrap(owner, method, label)
    return clock


def distribution(clock, seconds):
    if clock is None:
        return None
    assert not clock.stack
    rows = {key: dict(value) for key, value in clock.rows.items()}
    accounted = sum(row["exclusive_seconds"] for row in rows.values())
    assert accounted <= seconds + 0.001
    return {"rows": rows, "other_seconds": seconds - accounted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("baseline", "phases"), default="baseline")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--iterations",
        type=int,
        help="Bound training by official iterations instead of episodes; 0 measures fresh-policy rollout only",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        help="Measure a fixed number of EnvRunner steps instead of a whole episode",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--condition", default="normal")
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds < 1:
        parser.error("episodes and seconds must be positive")
    if args.iterations is not None and args.iterations < 0:
        parser.error("iterations must be nonnegative")
    if args.sample_steps is not None and args.sample_steps < 1:
        parser.error("sample-steps must be positive")
    source = args.source.resolve()
    output = args.output.resolve()
    if output.is_relative_to(source):
        parser.error("evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source / "src"))
    sys.dont_write_bytecode = True
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        RAY_USAGE_STATS_ENABLED="0",
        WANDB_MODE="disabled",
    )
    started = time.perf_counter()
    import ray
    import torch

    import smartsom
    from smartsom.config.base_diagnostic import generate_scenario, load_inputs
    from smartsom.config.codec import primitive
    from smartsom.learning.rllib_config import build_algorithm_config
    from smartsom.learning.rllib_metrics import FactoryProgress, factory_infos

    assert Path(smartsom.__file__).resolve().parent == source / "src/smartsom"
    torch.set_num_threads(1)
    ex, factory, workload, options, paths = load_inputs(args.experiment.resolve())
    if options.method not in {"ppo", "dqn"}:
        parser.error("requires an official RLlib learning recipe")
    if not ex.drain_frozen or ex.drain_ticks is None:
        parser.error("requires a frozen scientific drain; profiling never calibrates")
    scenario = generate_scenario(
        *ex.inputs_for(args.condition, factory, workload),
        args.seed,
        args.condition,
        ex.drain_ticks,
    )
    world = primitive(scenario)
    world_hash = hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()
    write(output / "world.json", world)
    episode_rows = []

    class ProfileProgress(FactoryProgress):
        def on_episode_end(self, *, episode, env_runner, metrics_logger, **kwargs):
            super().on_episode_end(
                episode=episode,
                env_runner=env_runner,
                metrics_logger=metrics_logger,
                **kwargs,
            )
            infos = factory_infos(episode, -1)
            values = [i["factory_metrics"] for i in infos if "factory_metrics" in i]
            if len(values) != 1:
                raise ValueError("episode lacks unique final factory metrics")
            row = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in values[0].items()
            }
            physical_tick = max(i["physical_tick"] for i in infos)
            row["physical_tick"] = (
                physical_tick.item()
                if hasattr(physical_tick, "item")
                else physical_tick
            )
            episode_rows.append(row)

    config = build_algorithm_config(
        scenario=scenario,
        options=options,
        algorithm=options.method,
        seed=args.seed,
        training=options.learning_policy.learner.params,
        env_runners={"num_env_runners": 0},
        learners={"num_learners": 0, "num_gpus_per_learner": 0},
    ).callbacks(ProfileProgress)
    write(output / "resolved-algorithm.json", config.to_dict())
    write(
        output / "scientific-inputs.json",
        {
            "experiment": ex.model_dump(mode="json", by_alias=True),
            "algorithm": options.model_dump(mode="json", by_alias=True),
        },
    )
    provenance = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source, text=True
        ),
        "implementation_sha256": digest_tree(source / "src/smartsom"),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phase_clock_sha256": hashlib.sha256(
            Path(__file__).with_name("phase_clock.py").read_bytes()
        ).hexdigest(),
        "world_sha256": world_hash,
        "seed": args.seed,
        "condition": args.condition,
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("ray", "torch", "numpy")
        },
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "operation": "bounded official training, then one fixed-world rollout",
        "requested_sample_steps": args.sample_steps,
        "requested_training_iterations": args.iterations,
        "requested_training_episodes": args.episodes
        if args.iterations is None
        else None,
        "initialization": "fresh seeded network, not a converged policy",
        "execution": "one local EnvRunner and local CPU learner; one numerical thread",
        "excluded": [
            "Tune/CLI orchestration",
            "tracking",
            "checkpoint I/O",
            "evaluation panel orchestration",
        ],
        "scientific_overrides": "one frozen world repeated; authored algorithm parameters retained",
    }
    write(output / "source.json", provenance)
    algorithm = None
    clock = None
    training_start = None
    rollout_start = None
    iterations = []
    status = {"status": "running", "mode": args.mode, "operation": "training"}
    write(output / "status.json", status)

    deadline_reached = False

    def expired(signum, frame):
        nonlocal deadline_reached
        deadline_reached = True
        raise TimeoutError(
            "bounded profiler deadline reached; do not extrapolate incomplete work"
        )

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    try:
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
            object_store_memory=512 * 1024**2,
        )
        algorithm = config.build_algo()
        startup = time.perf_counter() - started
        clock = install_phases(algorithm) if args.mode == "phases" else None
        training_start = time.perf_counter()
        while (
            len(episode_rows) < args.episodes
            if args.iterations is None
            else algorithm.iteration < args.iterations
        ):
            tick = time.perf_counter()
            result = algorithm.train()
            row = {
                "iteration": algorithm.iteration,
                "seconds": time.perf_counter() - tick,
                "physical_ticks": result.get("physical_ticks"),
                "episodes": result.get("episodes"),
                "learner_work": {
                    name: metrics.get("num_module_steps_trained_lifetime")
                    for name, metrics in result.get("learners", {}).items()
                },
                "timers": result.get("timers"),
            }
            iterations.append(row)
            write(
                output / "progress.json",
                {"iterations": iterations, "episodes": episode_rows},
            )
            print(
                json.dumps(
                    {
                        "stage": "training",
                        **{
                            k: row[k]
                            for k in (
                                "iteration",
                                "seconds",
                                "physical_ticks",
                                "episodes",
                            )
                        },
                    }
                ),
                flush=True,
            )
        training_seconds = time.perf_counter() - training_start
        training = {
            "wall_seconds": training_seconds,
            "iterations": iterations,
            "episode_rows": list(episode_rows),
            "phases": distribution(clock, training_seconds),
            "seconds_per_ended_episode": training_seconds / len(episode_rows)
            if episode_rows
            else None,
            "note": "includes learner updates and any partial next episode sampled by the final official iteration; ended includes truncation",
        }
        write(output / "training.json", training)
        status["operation"] = "rollout"
        write(output / "status.json", status)
        episode_rows.clear()
        runner = algorithm.env_runner
        case = {
            "case_id": "profile-fixed-world",
            "scenario": scenario,
            "action_seed": args.seed,
            "mode": "sample"
            if config.get_evaluation_config_object().explore
            else "argmax",
        }
        runner.env.unwrapped.call("set_evaluation_case", case)
        torch.manual_seed(args.seed)
        if clock:
            clock.rows.clear()
        rollout_start = time.perf_counter()
        episodes = runner.sample(
            **(
                {"num_timesteps": args.sample_steps}
                if args.sample_steps
                else {"num_episodes": 1}
            ),
            force_reset=True,
            explore=config.get_evaluation_config_object().explore,
        )
        rollout_seconds = time.perf_counter() - rollout_start
        if args.sample_steps is None:
            assert len(episodes) == len(episode_rows) == 1
        rollout = {
            "wall_seconds": rollout_seconds,
            "episodes": episode_rows,
            "env_steps": sum(ep.env_steps() for ep in episodes),
            "agent_steps": sum(ep.agent_steps() for ep in episodes),
            "phases": distribution(clock, rollout_seconds),
            "policy": case["mode"],
            "trained_iterations": algorithm.iteration,
            "note": "whole fixed-world episode; inference and environment only, no learner updates; trained_iterations=0 means fresh initialized policy",
        }
        if args.sample_steps is not None:
            rollout["note"] = "fixed EnvRunner step workload; not a whole-episode time"
            rollout["requested_steps"] = args.sample_steps
            rollout["inflight_worlds"] = [
                {
                    "physical_tick": env.unwrapped.episode.sim.tick,
                    **env.unwrapped.episode.metrics(),
                }
                for env in runner.env.unwrapped.envs
            ]
            # Outside the measured sampling interval: compare returned scientific
            # samples without process-specific episode IDs or timing fields.
            sample_hash = hashlib.sha256()

            def hash_value(value):
                import numpy as np

                if isinstance(value, dict):
                    for key in sorted(value):
                        hash_value(key)
                        hash_value(value[key])
                elif isinstance(value, (list, tuple)):
                    sample_hash.update(str(len(value)).encode() + b"[")
                    for item in value:
                        hash_value(item)
                elif isinstance(value, np.ndarray):
                    sample_hash.update(str((value.shape, value.dtype.str)).encode())
                    sample_hash.update(np.ascontiguousarray(value).data)
                else:
                    sample_hash.update(repr(value).encode() + b";")

            for episode in episodes:
                hash_value(
                    (
                        episode.get_observations(),
                        episode.get_actions(),
                        episode.get_rewards(),
                        episode.is_terminated,
                        episode.is_truncated,
                    )
                )
            rollout["sample_sha256"] = sample_hash.hexdigest()
        write(output / "rollout.json", rollout)
        status.update(
            status="completed",
            startup_seconds=startup,
            training_seconds=training_seconds,
            rollout_seconds=rollout_seconds,
            driver_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 if sys.platform != "darwin" else 1024**2),
        )
    except BaseException as error:
        status.update(
            status="failed", error=repr(error), deadline_reached=deadline_reached
        )
        if algorithm is not None:
            try:
                status["inflight_worlds"] = [
                    {
                        "physical_tick": env.unwrapped.episode.sim.tick,
                        "completed_orders": len(env.unwrapped.episode.sim.completed),
                        "total_orders": len(env.unwrapped.episode.sim.demands),
                        **env.unwrapped.episode.metrics(),
                    }
                    for env in algorithm.env_runner.env.unwrapped.envs
                ]
            except Exception as inspection_error:
                status["inflight_inspection_error"] = repr(inspection_error)
        if rollout_start is not None and status["operation"] == "rollout":
            elapsed = time.perf_counter() - rollout_start
            write(
                output / "partial-rollout.json",
                {
                    "interrupted": True,
                    "wall_seconds": elapsed,
                    "phases": distribution(clock, elapsed),
                    "inflight_worlds": status.get("inflight_worlds"),
                    "note": "incomplete sampling: hotspot evidence only, not whole-episode time",
                },
            )
        if training_start is not None and status["operation"] == "training":
            elapsed = time.perf_counter() - training_start
            write(
                output / "partial-training.json",
                {
                    "interrupted": True,
                    "wall_seconds": elapsed,
                    "completed_iterations": iterations,
                    "ended_episode_rows": episode_rows,
                    "phases": distribution(clock, elapsed),
                    "note": "incomplete official iteration: diagnostic hotspots only, not completed training throughput",
                },
            )
        raise
    finally:
        signal.alarm(0)
        cleanup_start = time.perf_counter()
        try:
            try:
                if algorithm is not None:
                    algorithm.stop()
            finally:
                ray.shutdown()
        except BaseException as cleanup_error:
            status.update(status="failed", cleanup_error=repr(cleanup_error))
            if "error" not in status:
                raise
        finally:
            status["cleanup_seconds"] = time.perf_counter() - cleanup_start
            status["end_to_end_seconds"] = time.perf_counter() - started
            write(output / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
