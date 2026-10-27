"""Complete-rollout transaction boundary shared by native and legacy Base training."""

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


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


class ReservationSession:
    def __init__(self, args):
        from smartsom.algorithms.base_policy import build_policy

        self.args = args
        self.root = Path(args.output)
        self.ex, self.factory, self.workload, self.options, paths = load_inputs(
            args.experiment
        )
        if not uses_learning(self.options):
            raise ValueError("train requires PPO algorithm input")
        self.drain = runtime.frozen_drain(args, paths)
        manifest = runtime.freeze(
            self.root,
            paths,
            "train",
            {
                "seed": args.seed,
                "drain_ticks": self.drain,
                "tick_budget": args.ticks or self.ex.train_ticks,
                "episode_budget": self.ex.train_episodes,
                "budget_boundary": self.ex.budget_boundary,
                "condition_sampling": self.ex.condition_sampling,
                "maximum_seconds": args.seconds,
                "deadline_unix": getattr(args, "deadline_unix", None),
                "calibration": json.loads(Path(args.calibration).read_text()),
            },
        )
        self.budget = manifest["tick_budget"]
        self.started = manifest["started_unix"]
        self.deadline = manifest.get("deadline_unix") or (
            self.started + manifest["maximum_seconds"]
        )
        self.policy = build_policy(self.options, args.seed)
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
            self.policy.load(generation / "model.pt")
        else:
            if (self.root / "last.pt").exists() or (
                self.root / "episodes.jsonl"
            ).exists():
                raise ValueError(
                    "old run lacks a continuation checkpoint; preserve it and start a new run"
                )
            self.commit()
        self.publish()
        self.validate()

    def commit(self):
        from uuid import uuid4

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

    def publish(self):
        marker = json.loads((self.checkpoints / "current.json").read_text())
        model = self.checkpoints / marker["generation"] / "model.pt"
        temporary = self.root / "last.pt.tmp"
        shutil.copyfile(model, temporary)
        temporary.replace(self.root / "last.pt")
        temporary = self.root / "episodes.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        temporary.replace(self.root / "episodes.jsonl")
        interval = self.ex.checkpoint_every_episodes
        if interval and len(self.rows) % interval == 0:
            periodic = self.root / "checkpoints"
            periodic.mkdir(exist_ok=True)
            target = periodic / f"episode-{len(self.rows):08d}.pt"
            if not target.exists():
                temporary = target.with_suffix(".tmp")
                shutil.copyfile(model, temporary)
                temporary.replace(target)

    def validate(self):
        from smartsom.experiments.reservation_validation import validate

        return validate(self)

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
            "status": status,
            "pid": os.getpid(),
            "physical_ticks": self.ticks,
            "tick_budget": self.budget if self.ex.train_episodes is None else None,
            "episode_budget": self.ex.train_episodes,
            "episodes": len(self.rows),
            "updates": self.policy.updates,
            "wall_seconds": time.time() - self.started,
            "deadline_unix": self.deadline,
            **(extra or {}),
        }
        runtime.write_json(self.root / "progress.json", result)
        return result

    def step(self):
        if self.budget_done() or time.time() >= self.deadline:
            return {**self.progress(), "done": True}
        condition = (
            self.rng.choice(runtime.CONDITIONS)
            if self.ex.condition_sampling == "random"
            else runtime.CONDITIONS[len(self.rows) % len(runtime.CONDITIONS)]
        )
        try:
            result, rewards, critic = runtime.episode(
                self.factory,
                self.workload,
                self.options,
                self.policy,
                seed_for(self.args.seed, "train", len(self.rows)) % (2**31),
                condition,
                self.drain,
                on_progress=self.progress,
                diagnostics=self.ex.run.diagnostics,
                wall_deadline=time.monotonic() + max(0, self.deadline - time.time()),
                max_ticks=(self.budget - self.ticks)
                if self.ex.train_episodes is None and self.ex.budget_boundary == "cut"
                else None,
            )
        except TimeoutError:
            self.policy.records = []
            return {**self.progress(), "done": True}
        self.policy.update(
            rewards, terminal=result["status"] == "completed", final_critic=critic
        )
        self.ticks += result["tick"]
        result.update(
            episode=len(self.rows) + 1,
            physical_ticks=self.ticks,
            updates=self.policy.updates,
            losses=self.policy.last_losses,
        )
        self.rows.append(result)
        self.commit()
        self.validate()
        progress = self.progress()
        return {**progress, "done": progress["status"] != "training"}


def train(args):
    session = ReservationSession(args)
    while not session.step()["done"]:
        pass
