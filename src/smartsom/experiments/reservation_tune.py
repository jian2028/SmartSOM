"""Optional local Ray adapter using the shared Base session and probe machinery."""

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from smartsom.experiments import reservation_runtime as runtime
from smartsom.experiments.control import boundary, requested
from smartsom.experiments.tuning_calibration import CalibrationController
from smartsom.experiments.tuning_probe import ProbeSupervisor, ProbeWork
from smartsom.experiments.tuning_resources import (
    ExecutionProfile,
    ResourceMonitor,
)


def probe(group, profile, remaining_seconds):
    import torch

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import load_inputs

    ex, factory, workload, options, _ = load_inputs(group["experiment"])
    policy = build_policy(options, ex.run.seed)
    torch.set_num_threads(profile.threads)
    start = time.monotonic()
    result, rewards, critic = runtime.episode(
        factory,
        workload,
        options,
        policy,
        ex.run.seed,
        "normal",
        group["drain"],
        max_ticks=ex.run.probe_ticks,
        metrics_level=ex.run.metrics,
        metric_interval=ex.run.metric_interval,
        wall_deadline=start + remaining_seconds,
    )
    rollout = time.monotonic() - start
    start = time.monotonic()
    policy.update(
        rewards, terminal=result["status"] == "completed", final_critic=critic
    )
    return ProbeWork(
        result["tick"],
        policy.updates,
        {"rollout": rollout, "update": time.monotonic() - start},
    )


def execute(root):
    # Importing/checking Base configurations never imports Ray.
    import ray
    from ray.tune import Callback, Trainable

    from smartsom.experiments.reservation import arguments, load
    from smartsom.experiments.reservation_session import ReservationSession
    from smartsom.experiments.tuning_broker import AdaptiveBroker
    from smartsom.experiments.tuning_ray import (
        _set_threads,
        build_tuner,
        local_runtime_environment,
        require_ray_contract,
        resource_dict,
    )

    require_ray_contract()
    root = Path(root).resolve()
    load(root)
    args, settings = arguments(root)
    from smartsom.config.base_diagnostic import load_inputs

    ex, _, _, _, paths = load_inputs(args.experiment)
    drain = runtime.frozen_drain(args, paths)
    monitor = ResourceMonitor()
    recommendation = root / "tuning.json"
    if not recommendation.exists():
        calibration_budget = root / "performance-budget.json"
        if not calibration_budget.exists():
            runtime.write_json(
                calibration_budget,
                {"deadline_unix": time.time() + settings.calibration_seconds},
            )
        remaining = (
            json.loads(calibration_budget.read_text())["deadline_unix"] - time.time()
        )
        if remaining <= 0:
            raise RuntimeError("original performance calibration deadline exhausted")
        report = CalibrationController(
            probe,
            monitor,
            active_limit=remaining,
            supervisor=ResumableProbeSupervisor(
                work_root=root / "probes",
                keep_artifacts=True,
                on_poll=lambda _progress: boundary(root),
            ),
        ).run(
            {
                "base": {
                    "experiment": str(args.experiment),
                    "drain": drain,
                    "runtime_device": "cpu",
                }
            },
            candidates={
                "base": [
                    ExecutionProfile(t, c)
                    for t in (1, 2)
                    for c in range(1, settings.max_concurrent + 1)
                ]
            },
            cancelled=lambda: requested(root),
        )
        runtime.write_json(root / "calibration-performance.json", asdict(report))
        if not report.ready:
            raise RuntimeError(
                "performance calibration incomplete; inspect calibration-performance.json"
            )
        selected = report.recommendations["base"]
        runtime.write_json(recommendation, asdict(selected))
    selected = ExecutionProfile(**json.loads(recommendation.read_text()))
    if settings.tune == "recommend":
        result = {"status": "recommended", "profile": asdict(selected)}
        runtime.write_json(root / "progress.json", result)
        return result

    budget_file = root / "tune-budget.json"
    if not budget_file.exists():
        runtime.write_json(
            budget_file, {"deadline_unix": time.time() + settings.seconds}
        )
    deadline = json.loads(budget_file.read_text())["deadline_unix"]

    class BaseTrainable(Trainable):
        def setup(self, config):
            load(config["root"])
            worker_args, _ = arguments(Path(config["root"]))
            worker_args.seed = config["seed"]
            worker_args.output = Path(config["root"]) / "seeds" / str(config["seed"])
            worker_args.deadline_unix = config["deadline"]
            self.session = ReservationSession(worker_args)
            self.threads = int(resource_dict(self.trial_resources)["CPU"])
            self.limits = _set_threads(self.threads)
            self.ack("training")
            from smartsom.experiments.evaluation_resume import durable_json

            ack = json.loads((self.session.root / "tuning-runtime.json").read_text())
            durable_json(
                self.session.root
                / "allocations"
                / f"{ack['allocation_epoch']}-{ack['pid']}.json",
                ack,
            )

        def ack(self, phase):
            import psutil

            runtime.write_json(
                self.session.root / "tuning-runtime.json",
                {
                    "schema": "smartsom.tune-worker/v1",
                    "experiment_id": self.config["experiment_id"],
                    "pid": os.getpid(),
                    "pid_create_time": psutil.Process().create_time(),
                    "threads": self.threads,
                    "allocation_epoch": self.config.get("execution_contract", {}).get(
                        "allocation_epoch", 0
                    ),
                    "phase": phase,
                    "child_processes": [],
                    "children_complete": True,
                    "physical_ticks": self.session.ticks,
                    "updates": self.session.policy.updates,
                },
            )

        def cleanup(self):
            if hasattr(self, "session") and hasattr(self, "threads"):
                self.ack("closed")
            if hasattr(self, "limits"):
                self.limits.restore_original_limits()

        def step(self):
            boundary(root)
            result = self.session.step()
            self.ack("training" if not result["done"] else "closed")
            return {**result, "experiment_id": self.config["experiment_id"]}

        def save_checkpoint(self, checkpoint_dir):
            # The native generation is authoritative; Ray checkpoints carry a verified pointer.
            marker = self.session.checkpoints / "current.json"
            shutil.copyfile(marker, Path(checkpoint_dir) / "native-generation.json")
            return checkpoint_dir

        def load_checkpoint(self, checkpoint_dir):
            # Worker setup already restored the newest atomic generation. Never roll
            # native progress back to an older Ray reporting checkpoint.
            saved = json.loads(
                (Path(checkpoint_dir) / "native-generation.json").read_text()
            )
            generation = self.session.checkpoints / saved["generation"]
            import hashlib

            if any(
                hashlib.sha256((generation / name).read_bytes()).hexdigest() != digest
                for name, digest in saved["files"].items()
            ):
                raise ValueError("Ray/native checkpoint identity mismatch")

    class StopCallback(Callback):
        def on_trial_start(self, iteration, trials, trial, **info):
            broker.note_actor(trial.config["experiment_id"], trial)

        def on_trial_result(self, iteration, trials, trial, result, **info):
            broker.note_actor(trial.config["experiment_id"], trial)

        def on_trial_error(self, iteration, trials, trial, **info):
            broker.actor_failed(trial.config["experiment_id"], trial)

        def on_experiment_end(self, trials, **info):
            broker.drain_shared()

        def on_step_begin(self, iteration, trials, **info):
            boundary(root)
            if time.time() >= deadline:
                raise TimeoutError("original Tune wall deadline reached")
            broker.refresh()
            runtime.write_json(root / "elasticity.json", broker.summary())
            rows = [
                json.loads(p.read_text())
                for p in sorted((root / "seeds").glob("*/progress.json"))
            ]
            runtime.write_json(
                root / "progress.json",
                {
                    "status": "training",
                    "stage": "tune",
                    "deadline_unix": deadline,
                    "physical_ticks": sum(row.get("physical_ticks", 0) for row in rows),
                    "tick_budget": ex.train_ticks * len(ex.train_seeds)
                    if ex.train_episodes is None
                    else None,
                    "episode_budget": ex.train_episodes * len(ex.train_seeds)
                    if ex.train_episodes is not None
                    else None,
                    "updates": sum(row.get("updates", 0) for row in rows),
                    "seeds": rows,
                },
            )

    # This adapter owns a local Ray runtime; it never attaches to somebody else's cluster.
    if ray.is_initialized():
        raise ValueError("Base Tune requires an owned local Ray runtime")
    measurements = json.loads((root / "calibration-performance.json").read_text())[
        "measurements"
    ]
    profiles = measured_profiles(measurements)
    if not profiles:
        raise RuntimeError("no valid measured CPU profiles")
    entries = []
    completed = []
    for seed in ex.train_seeds:
        directory = root / "seeds" / str(seed)
        progress = directory / "progress.json"
        previous = json.loads(progress.read_text()) if progress.exists() else {}
        if time.time() >= deadline and previous.get("status") != "completed":
            previous = {**previous, "status": "wall_budget_reached"}
        if previous.get("status") in {"completed", "wall_budget_reached"}:
            completed.append(
                {
                    "seed": seed,
                    "error": None,
                    **{
                        k: previous.get(k)
                        for k in ("physical_ticks", "episodes", "status")
                    },
                }
            )
            continue
        entries.append(
            {
                "experiment_id": str(seed),
                "root": str(root),
                "seed": seed,
                "run_dir": str(directory),
                "deadline": deadline,
                "task": "train",
                "remaining_updates": ex.train_ticks if ex.train_episodes is None else 0,
                "remaining_work_metric": "physical_ticks",
                "execution_contract": {"cpu_overhead": 0, "allocation_epoch": 0},
            }
        )
    if not entries:
        return finish(root, completed, selected)
    broker = AdaptiveBroker(
        entries,
        {e["experiment_id"]: profiles for e in entries},
        execution=settings.resource_execution,
        global_limit=settings.max_concurrent,
        control_root=root,
    )

    def allocation(controller, trial, result, scheduler):
        if ex.train_episodes is not None:
            completed = result.get("episodes", 0)
            ticks = result.get("physical_ticks", 0)
            # Resize economics use an explicitly estimated physical workload,
            # derived only from committed episodes. Session owns the exact budget.
            entry = broker.entries[trial.config["experiment_id"]]
            entry["remaining_updates"] = ticks + (
                max(0, ex.train_episodes - completed) * ticks / completed
                if completed
                else 0
            )
            entry["work_estimate"] = "committed-mean-episode-ticks"
        return broker.allocation(controller, trial, result, scheduler)

    initial = min(
        profiles, key=lambda p: (abs(p["CPU"] - selected.threads), -p["throughput"])
    )
    ray.init(
        address="local",
        num_cpus=max(p["CPU"] for p in profiles) * settings.max_concurrent,
        include_dashboard=False,
        runtime_env=local_runtime_environment(),
    )
    storage = root / "ray"
    # Reuse the project's new-segment/native-continuation contract, not stale Ray state.
    segment = f"base-{time.time_ns()}"
    try:
        tuner = build_tuner(
            entries,
            selected={
                e["experiment_id"]: {k: initial[k] for k in ("CPU", "GPU", "memory")}
                for e in entries
            },
            broker=broker,
            storage_path=storage,
            name=segment,
            device="cpu",
            allocation_function=allocation,
            callbacks=[StopCallback()],
            trainable=BaseTrainable,
            max_failures=settings.max_failures,
        )
        results = tuner.fit()
        boundary(root, fresh=True)
        rows = [
            {
                "seed": result.config["seed"],
                "error": str(result.error) if result.error else None,
                "physical_ticks": result.metrics.get("physical_ticks"),
                "episodes": result.metrics.get("episodes"),
                "status": result.metrics.get("status"),
            }
            for result in results
        ]
        rows.extend(completed)
        return finish(root, rows, selected)
    except TimeoutError:
        if time.time() < deadline:
            raise
        rows = []
        for seed in ex.train_seeds:
            path = root / "seeds" / str(seed) / "progress.json"
            progress = json.loads(path.read_text()) if path.exists() else {}
            rows.append(
                {
                    "seed": seed,
                    "error": None,
                    "physical_ticks": progress.get("physical_ticks", 0),
                    "episodes": progress.get("episodes", 0),
                    "status": "completed"
                    if progress.get("status") == "completed"
                    else "wall_budget_reached",
                }
            )
        return finish(root, rows, selected)
    finally:
        ray.shutdown()


def finish(root, rows, selected):
    result = {
        "status": (
            "failed"
            if any(r["error"] for r in rows)
            else "wall_budget_reached"
            if any(r["status"] == "wall_budget_reached" for r in rows)
            else "stopped"
            if any(r["status"] != "completed" for r in rows)
            else "completed"
        ),
        "seeds": rows,
        "profile": asdict(selected),
    }
    runtime.write_json(root / "summary.json", result)
    runtime.write_json(root / "progress.json", result)
    runtime.write_json(root / "batch.json", result)
    return result


def measured_profiles(measurements):
    """Per-learner rates/peaks from real probes; no invented update count forecast."""
    groups = {}
    for item in measurements:
        if not item["valid"] or item["throughput"] <= 0:
            continue
        p = item["profile"]
        if p["device"] != "cpu":
            continue
        stages = item["stages"]
        rate = item["throughput"] / p["concurrency"]
        memory = max(1, int(stages.get("per_trial_peak_memory", item["peak_memory"])))
        profile = {
            "CPU": p["threads"],
            "GPU": 0,
            "memory": memory,
            "concurrency": p["concurrency"],
            "throughput": rate,
            "update_seconds": 1 / rate,
            "restart_seconds": max(
                0,
                stages.get("original_elapsed_seconds", item["elapsed_seconds"])
                - stages.get("rollout", 0)
                - stages.get("update", 0),
            ),
        }
        groups.setdefault(p["threads"], []).append(profile)
    return [
        {
            "CPU": threads,
            "concurrency": max(p["concurrency"] for p in rows),
            "GPU": 0,
            "memory": max(p["memory"] for p in rows),
            "throughput": min(p["throughput"] for p in rows),
            "update_seconds": max(p["update_seconds"] for p in rows),
            "restart_seconds": max(p["restart_seconds"] for p in rows),
        }
        for threads, rows in groups.items()
    ]


class ResumableProbeSupervisor(ProbeSupervisor):
    """Replay verified completed measurements; retain the original wall deadline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indices = {}

    def run(self, probe, group, profile, remaining_seconds, cancelled=lambda: False):
        from smartsom.config.codec import digest
        from smartsom.experiments.evaluation_resume import durable_json
        from smartsom.experiments.tuning_calibration import CandidateMeasurement

        key = digest({"group": group, "profile": asdict(profile)})
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        path = self.work_root / "receipts" / f"{key}-{index}.json"
        if cancelled():
            return super().run(probe, group, profile, remaining_seconds, cancelled)
        if path.exists():
            receipt = json.loads(path.read_text())
            if digest(receipt["measurement"]) != receipt["sha256"]:
                raise ValueError("performance measurement receipt corrupted")
            data = receipt["measurement"]
            data["profile"] = ExecutionProfile(**data["profile"])
            data["stages"] = {
                **data["stages"],
                "reused_measurement": 1.0,
                "original_elapsed_seconds": data["elapsed_seconds"],
            }
            return CandidateMeasurement(**data)
        measurement = super().run(probe, group, profile, remaining_seconds, cancelled)
        if measurement.eligible:
            data = asdict(measurement)
            durable_json(path, {"measurement": data, "sha256": digest(data)})
        return measurement
