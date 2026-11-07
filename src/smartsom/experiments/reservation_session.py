"""Complete-rollout transaction boundary for current component training."""

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

from smartsom.algorithms.base_policy import uses_learning
from smartsom.config.base_diagnostic import load_inputs, seed_for
from smartsom.experiments import reservation_runtime as runtime
from smartsom.experiments.checkpoint_retention import _sync_directory, sync_tree
from smartsom.experiments.resource_guard import check as check_resources


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


class ReservationSession:
    def __init__(self, args):
        from smartsom.algorithms.base_policy import build_policy

        self.args = args
        self.root = Path(args.output)
        prepared = getattr(args, "_prepared_inputs", None)
        if prepared is None:
            inputs = load_inputs(args.experiment)
        else:
            inputs, expected = prepared
            if runtime.prepared_identity(inputs[-1]) != expected:
                raise ValueError("prepared input identity changed before training")
        self.ex, self.factory, self.workload, self.options, paths = inputs
        if not uses_learning(self.options):
            raise ValueError("train requires a configured learning policy")
        from smartsom.telemetry.evaluation_summary import training_context_identity

        self.drain = runtime.frozen_drain(args, paths)
        self.training_context = training_context_identity(
            self.ex.model_copy(
                update={
                    "drain_ticks": self.drain,
                    "train_ticks": args.ticks or self.ex.train_ticks,
                }
            ),
            self.factory,
            self.workload,
            self.options,
            paths,
        )
        manifest = runtime.freeze(
            self.root,
            paths,
            "train",
            {
                "seed": args.seed,
                "training_context": self.training_context,
                "tracking_group": getattr(args, "tracking_group", None),
                "experiment_short_name": getattr(args, "experiment_short_name", None),
                "drain_ticks": self.drain,
                "tick_budget": args.ticks or self.ex.train_ticks,
                "episode_budget": self.ex.train_episodes,
                "budget_boundary": self.ex.budget_boundary,
                "condition_sampling": self.ex.condition_sampling,
                "maximum_seconds": args.seconds,
                "deadline_unix": getattr(args, "deadline_unix", None),
                "calibration": json.loads(Path(args.calibration).read_text())
                if args.calibration
                else {"status": "explicit_frozen_drain", "drain_ticks": self.drain},
            },
        )
        self.budget = manifest["tick_budget"]
        if self.ex.validation.unit == "physical_ticks" and any(
            n > self.budget for n in self.ex.validation.milestones
        ):
            raise ValueError("validation node exceeds effective training tick budget")
        self.replay_source = manifest["source"]
        self.run_id = manifest["run_id"]
        self.source_sha256 = manifest["source"]["implementation_sha256"]
        self.started = manifest["started_unix"]
        self.deadline = manifest.get("deadline_unix") or (
            self.started + manifest["maximum_seconds"]
            if manifest["maximum_seconds"] is not None
            else float("inf")
        )
        from smartsom.telemetry.experiment_tracking import ExperimentTracking

        self.tracking = ExperimentTracking(
            self.root, self.ex.tracking, job_type="train"
        )
        check_resources("model_build", force=True)
        self.policy = build_policy(
            self.options,
            args.seed,
            numerical_threads=getattr(args, "numerical_threads", 1),
        )
        check_resources("model_built", force=True)
        self.rng = random.Random(seed_for(args.seed, "conditions"))
        self.rows = []
        self.ticks = 0
        self.checkpoints = self.root / "continuation"
        self.checkpoints.mkdir(exist_ok=True)
        pointer = self.checkpoints / "current.json"
        if pointer.exists():
            marker = json.loads(pointer.read_text())
            generation = self.checkpoints / marker["generation"]
            for name, digest in marker["files"].items():
                if (
                    hashlib.sha256((generation / name).read_bytes()).hexdigest()
                    != digest
                ):
                    raise ValueError("continuation checkpoint digest mismatch")
            state = json.loads((generation / "state.json").read_text())
            self.rows, self.ticks = state["rows"], state["ticks"]
            self.rng.setstate(tuples(state["condition_rng"]))
            check_resources("model_load", force=True)
            self.policy.load(generation / "model.pt")
            check_resources("model_loaded", force=True)
            self.publish()
        else:
            if (self.root / "last.pt").exists() or (
                self.root / "episodes.jsonl"
            ).exists():
                raise ValueError(
                    "old run lacks a continuation checkpoint; preserve it and start a new run"
                )
            self.commit()
        self.tracking.start(manifest, self.ex.model_dump(mode="json", by_alias=True))
        try:
            self.validate()
        except BaseException:
            self.close(failed=True)
            raise

    def commit(self):
        from uuid import uuid4

        check_resources("checkpoint_save", force=True)
        generation = self.checkpoints / f"{len(self.rows):08d}-{uuid4().hex}"
        generation.mkdir()
        self.policy.save(generation / "model.pt")
        runtime.write_json(
            generation / "state.json",
            {
                "rows": self.rows,
                "ticks": self.ticks,
                "condition_rng": self.rng.getstate(),
            },
        )
        sync_tree(generation)
        check_resources("checkpoint_publish", force=True)
        runtime.write_json(
            self.checkpoints / "current.json",
            {
                "generation": generation.name,
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in generation.iterdir()
                },
            },
        )
        with (self.checkpoints / "current.json").open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(self.checkpoints)
        self.publish()
        check_resources("checkpoint_committed", force=True)

    def publish(self):
        marker = json.loads((self.checkpoints / "current.json").read_text())
        model = self.checkpoints / marker["generation"] / "model.pt"
        committed = json.loads((model.parent / "state.json").read_text())
        evidence = {
            "schema": "smartsom.checkpoint-evidence/v1",
            "checkpoint_sha256": marker["files"]["model.pt"],
            "train_seed": self.args.seed,
            "episodes": len(committed["rows"]),
            "physical_ticks": committed["ticks"],
            "source_sha256": self.source_sha256,
            "run_id": self.run_id,
            "training_context": self.training_context,
        }
        temporary = self.root / "last.pt.tmp"
        shutil.copyfile(model, temporary)
        temporary.replace(self.root / "last.pt")
        runtime.write_json(self.root / "last.pt.json", evidence)
        temporary = self.root / "episodes.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        temporary.replace(self.root / "episodes.jsonl")
        interval = self.ex.checkpoint_every_episodes
        if (interval and len(self.rows) % interval == 0) or len(self.rows) in {
            c.episode for c in self.ex.evaluation.checkpoints
        }:
            periodic = self.root / "checkpoints"
            periodic.mkdir(exist_ok=True)
            target = periodic / f"episode-{len(self.rows):08d}.pt"
            if not target.exists():
                temporary = target.with_suffix(".tmp")
                shutil.copyfile(model, temporary)
                temporary.replace(target)
            runtime.write_json(target.with_suffix(".pt.json"), evidence)

    def validate(self):
        from smartsom.experiments.reservation_validation import validate

        if self.validation_node in self.ex.validation.milestones:
            self.progress({"status": "validating", "phase": "validation"})
        return validate(self)

    @property
    def validation_node(self):
        return (
            self.ticks
            if self.ex.validation.unit == "physical_ticks"
            else len(self.rows)
        )

    def rollout_limit(self):
        limit = (
            self.budget - self.ticks
            if self.ex.train_episodes is None and self.ex.budget_boundary == "cut"
            else None
        )
        if self.ex.validation.unit == "physical_ticks":
            upcoming = [
                n - self.ticks for n in self.ex.validation.milestones if n > self.ticks
            ]
            if upcoming:
                limit = min(limit, min(upcoming))
        return limit

    def budget_done(self):
        if self.ex.train_episodes is not None:
            return len(self.rows) >= self.ex.train_episodes
        return self.ticks >= self.budget

    def progress(self, extra=None):
        from smartsom.experiments.control import boundary

        boundary(self.root)
        status = (
            "completed"
            if self.budget_done()
            else ("wall_budget_reached" if time.time() >= self.deadline else "training")
        )
        result = {
            "phase": "train",
            "status": status,
            "pid": os.getpid(),
            "seed": self.args.seed,
            "run_id": self.run_id,
            "algorithm": self.options.method,
            "output_directory": str(self.root),
            "physical_ticks": self.ticks,
            "committed_physical_ticks": self.ticks,
            "tick_budget": self.budget if self.ex.train_episodes is None else None,
            "episode_budget": self.ex.train_episodes,
            "episodes": len(self.rows),
            "updates": self.policy.updates,
            "model_parameters": sum(
                p.numel() for p in self.policy.learner.models.parameters()
            )
            if hasattr(self.policy.learner, "models")
            else None,
            "network_count": len(self.policy.learner.models)
            if hasattr(self.policy.learner, "models")
            else None,
            "wall_seconds": time.time() - self.started,
            "deadline_unix": None if self.deadline == float("inf") else self.deadline,
            **(extra or {}),
            "tracking": dict(self.tracking.status),
        }
        from smartsom.telemetry.current_progress import presentation

        result, total, unit = presentation(result)
        result.update(total=total, unit=unit)
        self.tracking.log({"phase": "train", **result})
        runtime.write_json(self.root / "progress.json", result)
        return result

    def world_seed(self):
        seed = seed_for(
            self.args.seed,
            self.ex.training_world_domain,
            *self.ex.training_world_prefix,
            len(self.rows),
        )
        return (
            seed % self.ex.training_seed_modulus
            if self.ex.training_seed_modulus
            else seed
        )

    def final_evaluate(self):
        from types import SimpleNamespace

        import torch

        if not self.ex.final_evaluation or not self.budget_done():
            return
        checkpoint = self.root / (
            "best.pt" if (self.root / "best.pt").exists() else "last.pt"
        )
        plans = (
            [
                (
                    self.root / "checkpoints" / f"episode-{c.episode:08d}.pt",
                    self.root / "evaluation" / f"episode-{c.episode:08d}",
                    c.modes,
                )
                for c in self.ex.evaluation.checkpoints
            ]
            if self.ex.evaluation.checkpoints
            else [(checkpoint, self.root / "evaluation", self.ex.validation.modes)]
        )
        panels = []
        with torch.random.fork_rng(devices=[]):
            for model, directory, modes in plans:
                if not model.is_file():
                    raise ValueError(
                        f"missing prespecified evaluation checkpoint: {model}"
                    )
                evidence = json.loads(model.with_suffix(".pt.json").read_text())
                for mode in modes:
                    runtime.evaluate(
                        SimpleNamespace(
                            experiment=self.args.experiment,
                            tracking=self.tracking,
                            output=directory / mode,
                            native=True,
                            calibration=self.args.calibration,
                            checkpoint=str(model),
                            method=["configured"],
                            selection=mode,
                            split="test",
                            cases=self.ex.test_cases,
                            action_seeds=self.ex.validation.sample_action_seeds
                            if mode == "sample"
                            else 1,
                            seconds=self.args.seconds,
                            deadline_unix=None
                            if self.deadline == float("inf")
                            else self.deadline,
                            numerical_threads=torch.get_num_threads(),
                            training_progress={
                                "train_seed": self.args.seed,
                                "episodes": evidence["episodes"],
                                "physical_ticks": evidence["physical_ticks"],
                            },
                        )
                    )
                    panels.append(
                        {
                            "checkpoint": str(model.relative_to(self.root)),
                            "selection": mode,
                        }
                    )
        runtime.write_json(
            self.root / "evaluation/completed.json",
            {
                "modes": list(self.ex.validation.modes),
                "checkpoint": checkpoint.name,
                "panels": panels,
            },
        )
        self.progress()

    def step(self):
        if self.budget_done() or time.time() >= self.deadline:
            return {**self.progress(), "done": True}
        condition = (
            self.rng.choice(self.ex.training_conditions)
            if self.ex.condition_sampling == "random"
            else self.ex.training_conditions[
                len(self.rows) % len(self.ex.training_conditions)
            ]
        )
        try:
            check_resources("train_rollout", force=True)
            result, rewards, critic = runtime.episode(
                *self.ex.inputs_for(condition, self.factory, self.workload),
                self.options,
                self.policy,
                self.world_seed(),
                condition,
                self.drain,
                on_progress=self.progress,
                progress_interval=self.ex.display.summary_interval,
                diagnostics=self.ex.run.diagnostics,
                metrics_level=self.ex.run.metrics,
                metric_interval=self.ex.run.metric_interval,
                window_analysis=self.ex.evaluation.window_analysis,
                wall_deadline=time.monotonic() + max(0, self.deadline - time.time()),
                max_ticks=self.rollout_limit(),
                replay_directory=(
                    self.root
                    / "recordings"
                    / "train"
                    / f"episode-{len(self.rows) + 1:08d}"
                    if self.ex.replay.train == "all"
                    or (
                        self.ex.replay.train == "first"
                        and not any(row["condition"] == condition for row in self.rows)
                    )
                    else None
                ),
                replay_source=self.replay_source,
            )
        except TimeoutError:
            self.policy.discard_rollout()
            return {**self.progress(), "done": True}
        update_started = time.monotonic()
        check_resources("learner_update", force=True)
        self.policy.update(
            rewards, terminal=result["status"] == "completed", final_critic=critic
        )
        check_resources("learner_updated", force=True)
        result["learner_update_seconds"] = time.monotonic() - update_started
        if self.options.method == "ppo":
            result["ppo_update_seconds"] = result["learner_update_seconds"]
        self.ticks += result["tick"]
        result.update(
            episode=len(self.rows) + 1,
            episodes=len(self.rows) + 1,
            physical_ticks=self.ticks,
            tick_budget=self.budget if self.ex.train_episodes is None else None,
            episode_budget=self.ex.train_episodes,
            updates=self.policy.updates,
            losses=self.policy.last_losses,
        )
        self.rows.append(result)
        self.commit()
        self.tracking.log({"phase": "train", **result})
        self.validate()
        progress = self.progress()
        return {**progress, "done": progress["status"] != "training"}

    def close(self, *, failed=False):
        self.tracking.close(failed=failed)


def train(args):
    session = ReservationSession(args)
    try:
        while not session.step()["done"]:
            pass
        session.final_evaluate()
    except BaseException:
        session.close(failed=True)
        raise
    else:
        session.close()
