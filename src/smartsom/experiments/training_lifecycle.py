"""Opt-in update-point recovery, fixed validation, and checkpoint references.

Only this coordinator writes lifecycle evidence. Backends retain their PPO loops;
their adapters supply state at a completed optimizer update, never mid-rollout.
"""

import json
import shutil
import signal
import threading
from dataclasses import asdict
from pathlib import Path

from smartsom.config.codec import digest, primitive
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.training_controls import TrainingControls
from smartsom.experiments.training_validation import (
    evaluate_inputs,
    predictor,
    select_best,
    validation_inputs,
)
from smartsom.learning.checkpoint import (
    CheckpointFile,
    CheckpointManifest,
    ResourceCheckpointManifest,
    RoleWeights,
    file_hash,
    structural_identity,
)

SCHEMA = "smartsom.update-checkpoint/v1"
_COUNTERS = (
    "sampled_steps",
    "updates",
    "completed",
    "failed",
    "agent_steps",
    "physical_actions",
    "conflicts",
    "recorded",
)


def _members(directory):
    return {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p != directory / "manifest.json"
        and not p.relative_to(directory).parts[0] == "references"
    }


def inspect_resume_checkpoint(path):
    """Verify all members before deserializing a local trusted framework artifact."""
    directory = Path(path).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("not a complete resumable update checkpoint")
    if manifest["files"] != _members(directory):
        raise ValueError("resumable checkpoint member digests disagree")
    return manifest


def load_resumable_training(path):
    from smartsom.experiments.training_audit import load_training_snapshot

    inspect_resume_checkpoint(path)
    return load_training_snapshot(Path(path) / "resolved_training.json")


def protect_checkpoint(path, reference):
    """Register an explicit report/reference so automatic retention cannot delete it."""
    inspect_resume_checkpoint(path)
    directory = Path(path) / "references"
    directory.mkdir(exist_ok=True)
    write_json(
        directory / f"{digest(str(reference))}.json", {"reference": str(reference)}
    )


def _identity(resolved, controls):
    source = source_identity()
    root = Path(__file__).resolve().parents[3]
    content = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted((root / "src").rglob("*.py"))
    }
    content.update(
        {name: file_hash(root / name) for name in ("pyproject.toml", "uv.lock")}
    )
    # Paths and recording destinations do not change the scientific training recipe.
    recipe = primitive(resolved)
    recipe.pop("sources", None)
    recipe["run"].pop("output_root", None)
    recipe["run"].pop("algorithm", None)
    recipe["run"].pop("scenario", None)
    recipe["base"].pop("sources", None)
    recipe["base"]["run"].pop("output_root", None)
    return {
        "source_commit": source["git"]["commit"],
        "source_sha256": digest(content),
        "python": source["python"],
        "platform": source["platform"],
        "dependencies": source["packages"],
        "recipe_sha256": digest(recipe),
        "device": controls.device,
        "numerical_threads": controls.numerical_threads,
        "num_envs": controls.num_envs,
        "sampling_processes": controls.sampling_processes,
        "validation": primitive(controls.validation),
    }


class TrainingLifecycle:
    def __init__(self, resolved, controls):
        if not isinstance(controls, TrainingControls):
            raise TypeError("controls requires TrainingControls")
        self.resolved, self.controls = resolved, controls
        if resolved.algorithm.algorithm.parameters.n_steps % controls.num_envs:
            raise ValueError("global update quota must divide evenly across num_envs")
        if controls.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable on this host")
        self.identity = _identity(resolved, controls)
        self.resume_manifest = None
        if controls.resume_from:
            self.resume_manifest = inspect_resume_checkpoint(controls.resume_from)
            if self.resume_manifest["identity"] != self.identity:
                raise ValueError(
                    "complete resume requires identical source, recipe, dependencies and device"
                )
            if (
                self.resume_manifest["environment_steps"]
                >= resolved.run.budget.environment_steps
            ):
                raise ValueError(
                    "checkpoint has exhausted its original training budget; use independent weights initialization"
                )
            if (
                controls.stop_after_updates is not None
                and controls.stop_after_updates <= self.resume_manifest["ppo_updates"]
            ):
                raise ValueError("boundary stop must be after the resumed update point")
        self.status, self.stopped = "completed", False
        self.stop_requested, self.interrupts = False, 0
        self.ppo_updates = 0
        self.best_score, self.best_checkpoint, self.last_checkpoint = None, None, None
        self.no_improvement = 0
        self.adapter = self.evidence = None
        self.initial = None
        self._previous_signal = None
        self.inputs = (
            validation_inputs(resolved, controls.validation)
            if controls.validation
            else ()
        )

    def bind(self, evidence, stack):
        self.evidence = evidence
        self.root = evidence.run_dir / "checkpoints"
        self.root.mkdir()
        evidence.num_envs = self.controls.num_envs
        if self.controls.num_envs > 1:
            write_json(
                evidence.run_dir / "training_streams.json",
                {
                    "schema": "smartsom.training-streams/v1",
                    "num_envs": self.controls.num_envs,
                    "sampling_processes": self.controls.sampling_processes,
                    "episode_index": "local_episode * num_envs + stream_id",
                    "collection_order": "vector_tick_then_stream_id",
                    "steps_per_update": self.resolved.algorithm.algorithm.parameters.n_steps,
                },
            )
        write_json(evidence.run_dir / "training_controls.json", asdict(self.controls))
        if self.inputs:
            write_json(evidence.run_dir / "validation_inputs.json", self.inputs)
        if threading.current_thread() is threading.main_thread():
            self._previous_signal = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._interrupt)
            stack.callback(signal.signal, signal.SIGINT, self._previous_signal)

    def _interrupt(self, *_):
        self.interrupts += 1
        if self.interrupts > 1:
            raise KeyboardInterrupt
        self.stop_requested = True

    def attach(self, adapter):
        from smartsom.learning.training_state import (
            isolated_rng,
            load_state,
            restore_rng,
        )

        self.adapter = adapter
        controls, evidence = self.controls, self.evidence
        if controls.resume_from:
            directory = controls.resume_from
            state = load_state(directory / "session.pkl")
            for key, value in state["counters"].items():
                setattr(evidence, key, value)
            for name, stream in (
                ("episodes.jsonl", evidence.ledger),
                ("learner_metrics.jsonl", evidence.metrics),
            ):
                stream.write((directory / name).read_text())
                stream.flush()
            for item in (directory / "failures").glob("*.json"):
                shutil.copy2(item, evidence.run_dir / item.name)
            self.ppo_updates = state["ppo_updates"]
            self.initial, self.best_score = state["initial"], state["best_score"]
            self.no_improvement = state["no_improvement"]
            self.best_checkpoint = (
                Path(state["best_checkpoint"]) if state["best_checkpoint"] else None
            )
            if self.best_checkpoint:
                protect_checkpoint(self.best_checkpoint, evidence.run_dir)
                if controls.save_best:
                    write_json(
                        self.root / "best.json",
                        {"checkpoint": str(self.best_checkpoint)},
                    )
            adapter.restore(directory / "training")
            restore_rng(load_state(directory / "rng.pkl"))
        else:
            if controls.initialize_from:
                directory = controls.initialize_from
                if (directory / "manifest.json").exists():
                    inspect_resume_checkpoint(directory)
                    directory = directory / "inference"
                metadata = json.loads((directory / "checkpoint.json").read_text())
                spec = self.resolved.algorithm.algorithm
                if metadata["provider"] != spec.provider or metadata[
                    "structure_sha256"
                ] != structural_identity(self.resolved.base, spec.projection):
                    raise ValueError(
                        "weights initialization provider or input/action structure differs"
                    )
                for member in metadata["files"]:
                    path = (directory / member["path"]).resolve()
                    if (
                        not path.is_relative_to(directory.resolve())
                        or file_hash(path) != member["sha256"]
                    ):
                        raise ValueError(
                            "weights initialization member digest mismatch"
                        )
                with isolated_rng():
                    adapter.initialize(directory)
            self.initial = adapter.hashes()
        return self.initial

    def _inference_manifest(self, directory):
        spec, evidence = self.resolved.algorithm.algorithm, self.evidence
        final = self.adapter.hashes()
        resource = evidence.resource
        metadata = dict(
            schema="smartsom.resource-checkpoint/v1"
            if resource
            else "smartsom.checkpoint/v1",
            provider=spec.provider,
            projection=spec.projection,
            parameters=spec.parameters,
            structure_sha256=structural_identity(self.resolved.base, spec.projection),
            files=tuple(
                CheckpointFile(path=str(p.relative_to(directory)), sha256=file_hash(p))
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            ),
            dependencies=tuple(sorted(self.resume_dependencies.items())),
            initial_weights_sha256=digest(self.initial)
            if resource
            else next(iter(self.initial.values())),
            final_weights_sha256=digest(final)
            if resource
            else next(iter(final.values())),
            environment_steps=evidence.sampled_steps,
            learner_updates=evidence.updates,
            framework_seed=self.resolved.framework_seed,
        )
        if resource:
            active = evidence.active_env
            metadata.update(
                role_mapping=tuple(
                    (a, active.policy_for_agent(a)) for a in active.possible_agents
                ),
                role_weights=tuple(
                    RoleWeights(
                        role=r, initial_sha256=self.initial[r], final_sha256=final[r]
                    )
                    for r in sorted(final)
                ),
                agent_steps=evidence.agent_steps,
                physical_actions=evidence.physical_actions,
            )
        write_json(
            directory / "checkpoint.json",
            (ResourceCheckpointManifest if resource else CheckpointManifest)(
                **metadata
            ),
        )

    def after_update(self, metrics):
        from smartsom.learning.training_state import dump_state, isolated_rng, rng_state

        evidence, controls = self.evidence, self.controls
        self.ppo_updates = (
            evidence.sampled_steps
            // self.resolved.algorithm.algorithm.parameters.n_steps
        )
        if evidence.on_progress:
            evidence.on_progress(
                {
                    "stage": "learning_metrics",
                    "sampled_steps": evidence.sampled_steps,
                    "learner_updates": evidence.updates,
                    "ppo_updates": self.ppo_updates,
                    "metrics": metrics,
                }
            )
        budget_end = (
            evidence.sampled_steps == self.resolved.run.budget.environment_steps
        )
        explicit_stop = (
            controls.stop_after_updates is not None
            and self.ppo_updates >= controls.stop_after_updates
        )
        stop = self.stop_requested or explicit_stop
        if stop:
            self.stopped, self.status = True, "interrupted"
        validate = (
            controls.validation
            and self.ppo_updates % controls.validation.every_updates == 0
        )
        save_last = controls.save_last and (
            budget_end
            or stop
            or controls.checkpoint_every_updates is not None
            and self.ppo_updates % controls.checkpoint_every_updates == 0
        )
        if not validate and not save_last:
            return
        directory = self.root / f"update-{self.ppo_updates:06d}"
        directory.mkdir()
        selected = False
        with isolated_rng():
            saved_rng = rng_state()
            self.adapter.export(directory / "inference")
            self._inference_manifest(directory / "inference")
            if validate:
                settings = controls.validation
                report = evaluate_inputs(
                    self.resolved,
                    settings,
                    self.inputs,
                    predictor(
                        self.resolved.algorithm.algorithm.provider,
                        directory / "inference",
                        self.adapter.hashes(),
                        deterministic=settings.deterministic,
                        seed=settings.seed,
                    ),
                )
                selected, reason = select_best(report, self.best_score, settings)
                report.update(
                    ppo_updates=self.ppo_updates,
                    selection_reason=reason,
                    selected=selected,
                    weights=self.adapter.hashes(),
                )
                write_json(
                    evidence.run_dir / f"validation-{self.ppo_updates:06d}.json", report
                )
                decision = None
                if evidence.on_progress:
                    decision = evidence.on_progress(
                        {
                            "stage": "validation",
                            "report": primitive(report),
                            "ppo_updates": self.ppo_updates,
                            "sampled_steps": evidence.sampled_steps,
                        }
                    )
                if selected:
                    self.best_score, self.no_improvement = report, 0
                    if controls.save_best:
                        self.best_checkpoint = directory
                else:
                    self.no_improvement += 1
                if (
                    settings.patience is not None
                    and self.no_improvement >= settings.patience
                ):
                    stop, self.status = True, "early_stopped"
                    save_last = controls.save_last
                if isinstance(decision, dict) and decision.get("stop") == "pruned":
                    stop, self.status = True, "pruned"
                    save_last = controls.save_last
            if stop:
                self.stopped = True
                if self.status not in ("early_stopped", "pruned"):
                    self.status = "interrupted"
            if save_last or selected and controls.save_best:
                self.adapter.save(directory / "training")
                from smartsom.learning.state_audit import checkpoint_state_summary

                state_summary = checkpoint_state_summary(
                    self.resolved.algorithm.algorithm.provider, directory / "training"
                )
                if (
                    state_summary["environment_steps"] != evidence.sampled_steps
                    or state_summary["learner_updates"] != evidence.updates
                ):
                    raise ValueError(
                        "serialized framework counters differ from the completed update"
                    )
                write_json(directory / "state_summary.json", state_summary)
                dump_state(directory / "rng.pkl", saved_rng)
                dump_state(
                    directory / "session.pkl",
                    {
                        "counters": {key: getattr(evidence, key) for key in _COUNTERS},
                        "ppo_updates": self.ppo_updates,
                        "initial": self.initial,
                        "best_score": self.best_score,
                        "no_improvement": self.no_improvement,
                        "best_checkpoint": str(self.best_checkpoint)
                        if self.best_checkpoint
                        else None,
                    },
                )
                write_json(
                    directory / "resolved_training.json",
                    {
                        "schema": "smartsom.resolved-training/v1",
                        "resolved": self.resolved,
                    },
                )
                for name, stream in (
                    ("episodes.jsonl", evidence.ledger),
                    ("learner_metrics.jsonl", evidence.metrics),
                ):
                    stream.flush()
                    shutil.copy2(evidence.run_dir / name, directory / name)
                failures = directory / "failures"
                failures.mkdir()
                for item in evidence.run_dir.glob("episode_*_failure.json"):
                    shutil.copy2(item, failures / item.name)
                write_json(
                    directory / "manifest.json",
                    {
                        "schema": SCHEMA,
                        "status": "complete",
                        "identity": self.identity,
                        "ppo_updates": self.ppo_updates,
                        "environment_steps": evidence.sampled_steps,
                        "files": _members(directory),
                    },
                )
                if save_last:
                    self.last_checkpoint = directory
                    write_json(self.root / "last.json", {"checkpoint": str(directory)})
                if selected and controls.save_best:
                    write_json(self.root / "best.json", {"checkpoint": str(directory)})
            else:
                shutil.rmtree(directory)
        self._retain()

    def _retain(self):
        directories = sorted(
            p for p in self.root.glob("update-*") if (p / "manifest.json").exists()
        )
        recent = set(directories[-self.controls.keep_last :])
        protected = recent | {self.last_checkpoint, self.best_checkpoint}
        for directory in directories:
            if directory not in protected and not (directory / "references").exists():
                shutil.rmtree(directory)
