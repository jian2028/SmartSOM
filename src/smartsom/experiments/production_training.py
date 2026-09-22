"""Grid episodes within the public experiment and update-checkpoint lifecycle."""

import json
import math
import shutil
import signal
import threading
import time
from pathlib import Path
from statistics import mean

from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.experiment import ExperimentConfig, PreparedExperiment
from smartsom.config.production import ProductionRecipe, recipe_identity
from smartsom.config.training import episode_root
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.training_validation import select_best
from smartsom.learning.checkpoint import file_hash
from smartsom.telemetry.runtime import backend_diagnostics, bind, emit, operation


class ProductionEvidence:
    """Cumulative episode and learner ledgers owned by the coordinator."""

    def __init__(self, directory, previous=None, *, resource=False, every_seconds=0.5):
        self.run_dir = directory
        self.resource, self.every_seconds = resource, every_seconds
        self.completed = self.failed = 0
        self.sampled_steps = self.updates = self.agent_steps = self.physical_actions = (
            self.conflicts
        ) = 0
        self.active_envs = []
        self.active_env = None
        self.previous = previous
        self.recorded = set()
        self.on_progress = None
        self.last_progress = 0.0
        self.finished_ticks = 0
        self.finished_deliveries = 0
        self.latest_episode = {}

    def bind(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("episodes.jsonl", "learner_metrics.jsonl"):
            target = self.run_dir / name
            if self.previous and (self.previous / name).is_file():
                shutil.copyfile(self.previous / name, target)
            elif not target.exists():
                target.touch()
        rows = [
            json.loads(line)
            for line in (self.run_dir / "episodes.jsonl").read_text().splitlines()
        ]
        self.finished_ticks = sum(
            row["trace"][-1]["tick"] if row["trace"] else 0 for row in rows
        )
        # Older evidence has no final delivery count: do not reconstruct one from success.
        if rows:
            self.finished_deliveries = None
            self.latest_episode = {
                "episode_return": rows[-1]["return"],
                "episode_makespan": rows[-1]["makespan"],
            }
        self.recorded = {row["episode"] for row in rows}
        self.completed = sum(row["end_reason"] == "completed" for row in rows)
        self.failed = len(rows) - self.completed
        if self.previous and (self.previous / "active_episodes.json").is_file():
            rows.extend(
                json.loads((self.previous / "active_episodes.json").read_text())
            )
        steps = [step for row in rows for step in row["steps"]]
        if self.resource:
            self.agent_steps = sum(len(step["indices"]) for step in steps)
            self.physical_actions = sum(len(step["actions"]) for step in steps)
            self.conflicts = sum(step.get("conflict_count", 0) for step in steps)

    def append(self, name, data):
        with (self.run_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(data) + "\n")

    @staticmethod
    def row(snapshot):
        return {
            "episode": snapshot.episode_index,
            "stream_id": snapshot.stream_id,
            "local_episode": snapshot.local_episode,
            "root_seed": snapshot.input.seed,
            "input_sha256": digest(snapshot.input),
            "end_reason": snapshot.reason,
            "return": snapshot.total_reward,
            "makespan": snapshot.result["tick"]
            if snapshot.reason == "completed"
            else None,
            "steps": primitive(snapshot.steps),
            "extension_state": snapshot.extension_state,
            "trace": [
                {
                    "actions": row["actions"],
                    "tick": row["tick"],
                    "effects_sha256": digest(
                        {
                            key: row[key]
                            for key in ("state", "events", "reward", "rejections")
                        }
                    ),
                }
                for row in snapshot.trace
            ],
        }

    def episode(self, snapshot):
        if snapshot.episode_index in self.recorded:
            return
        if snapshot.result is not None:
            self.finished_ticks += snapshot.result["tick"]
            deliveries = len(snapshot.result["completed"])
            if self.finished_deliveries is not None:
                self.finished_deliveries += deliveries
            self.latest_episode = {
                "episode_return": snapshot.total_reward,
                "episode_deliveries": deliveries,
                "episode_makespan": snapshot.result["tick"]
                if snapshot.reason == "completed"
                else None,
            }
        self.recorded.add(snapshot.episode_index)
        self.completed += snapshot.reason == "completed"
        self.failed += snapshot.reason != "completed"
        self.append("episodes.jsonl", self.row(snapshot))

    def presentation(self):
        active = [
            s
            for s in self.active_envs
            if s is not None
            and s.episode_index not in self.recorded
            and s.result is not None
        ]
        return {
            "physical_ticks": self.finished_ticks
            + sum(s.result["tick"] for s in active),
            "training_deliveries": None
            if self.finished_deliveries is None
            else self.finished_deliveries
            + sum(len(s.result["completed"]) for s in active),
            **self.latest_episode,
        }

    def save_active(self, directory):
        write_json(
            directory / "active_episodes.json",
            [
                self.row(snapshot)
                for snapshot in self.active_envs
                if snapshot is not None and snapshot.episode_index not in self.recorded
            ],
        )

    def progress(self, stage, *, force=False):
        if self.on_progress and (
            force or time.monotonic() - self.last_progress >= self.every_seconds
        ):
            self.last_progress = time.monotonic()
            self.on_progress(
                {
                    "stage": stage,
                    "sampled_steps": self.sampled_steps,
                    "learner_updates": self.updates,
                    "completed_episodes": self.completed,
                    "failed_episodes": self.failed,
                    **(
                        {
                            "agent_steps": self.agent_steps,
                            "physical_actions": self.physical_actions,
                            "conflicts": self.conflicts,
                        }
                        if self.resource
                        else {}
                    ),
                }
            )


def identity(prepared, config):
    root = Path(__file__).resolve().parents[3]
    source = source_identity()
    paths = sorted((root / "src").rglob("*.py"))
    paths += [
        root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file()
    ]
    return {
        "recipe": digest(
            recipe_identity(prepared.resolved, config, prepared.validation_json)
        ),
        "source": digest({str(p.relative_to(root)): file_hash(p) for p in paths}),
        "source_commit": source["git"]["commit"],
        "python": source["python"],
        "platform": source["platform"],
        "dependencies": source["packages"],
        "runtime": primitive(config.runtime),
        "validation_controls": primitive(config.validation),
        "checkpoint_controls": primitive(config.checkpointing),
    }


def request_interrupt(state, *_):
    """First SIGINT stops at a checkpoint boundary; the second interrupts now."""
    if state["interrupted"]:
        raise KeyboardInterrupt
    state["interrupted"] = True


def retain_checkpoints(root, state, keep_last):
    """Keep latest, best, and immutable exports referenced by evaluations."""
    retained = sorted((root / "checkpoints").glob("update-*"))
    protected = {root / state[key] for key in ("last", "best") if state[key]}
    protected.update(retained[-keep_last:])
    for old in retained:
        if old not in protected and not (old.parent / "references" / old.name).exists():
            shutil.rmtree(old)


def verify_checkpoint(path):
    path = Path(path)
    data = json.loads((path / "update.json").read_text())
    actual = {
        str(p.relative_to(path)): file_hash(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "update.json"
    }
    if actual != data["files"]:
        raise ValueError("update checkpoint member digests disagree")
    return data


def validation_report(recipe, config, checkpoint, frozen_cases):
    from smartsom.config.validation import validate_frozen_cases
    from smartsom.experiments.training_validation import validation_inputs
    from smartsom.learning.production import LearnedProductionDriver
    from smartsom.trace.production import ExecutionAudit

    cases = (
        validate_frozen_cases(frozen_cases, config.validation)
        if frozen_cases
        else validation_inputs(recipe, config.validation)
    )
    rows = []
    for item in cases:
        case_id, replication = item["case_id"], item["replication"]
        seed, scenario = item["world_seed"], item["episode"]
        item.setdefault("input_id", digest([case_id, seed, replication, scenario]))
        driver = LearnedProductionDriver(
            checkpoint,
            scenario,
            deterministic=config.validation.deterministic,
            seed=seed,
        )
        try:
            checker = (
                ExecutionAudit(
                    scenario, driver.sim.snapshot(), driver.learning_contract
                )
                if config.validation.full_replay
                else None
            )
            while not driver.env.finished:
                row = driver.next_tick()
                if checker and row is not None:
                    checker.append(row)
            if checker:
                checker.finish_pending(driver.env.pending_decisions)
            sim = driver.sim
            rows.append(
                {
                    "input_id": item["input_id"],
                    "world_seed": seed,
                    "world_sha256": digest(scenario),
                    "case_id": case_id,
                    "replication": replication,
                    "reason": driver.env.reason,
                    "return": sim.total_reward,
                    "makespan": sim.tick if sim.status == "completed" else None,
                    "passing_rate": len(sim.completed) / max(1, len(sim.demands)),
                    "replay": "passed" if checker else "not_requested",
                }
            )
        finally:
            driver.env.close()
    successes = [r for r in rows if r["reason"] == "completed"]
    return {
        "episodes": len(rows),
        "weights": json.loads((checkpoint / "checkpoint.json").read_text())["weights"],
        "completed": len(successes),
        "results": rows,
        "successful_inputs": sorted(r["input_id"] for r in successes),
        "metrics": {
            name: mean(r[name] for r in successes) if successes else None
            for name in ("makespan", "return", "passing_rate")
        },
    }


@operation("training")
def train_prepared(
    prepared, *, initialize_from=None, on_progress=None, root=None, resume_from=None
):
    from smartsom.api import TrainingResult, _allocate, _finish
    from smartsom.experiments.packaging import model_locator
    from smartsom.learning.training_state import isolated_rng
    from smartsom.telemetry.training import TrainingDisplay

    config = ExperimentConfig.model_validate_json(prepared.config_json)
    recipe = prepared.resolved
    if json.loads(recipe.training_json) != primitive(config.training):
        raise ValueError("frozen recipe disagrees with recorded training budget")
    if bool(config.validation.enabled and config.validation.scenarios) != bool(
        prepared.validation_json
    ):
        raise ValueError("external validation cases require frozen input snapshots")
    if (
        digest(recipe_identity(recipe, config, prepared.validation_json))
        != prepared.scientific_sha256
    ):
        raise ValueError("frozen training input identity mismatch")
    signature = identity(prepared, config)
    restored = verify_checkpoint(resume_from) if resume_from else None
    if restored and restored["identity"] != signature:
        raise ValueError("resume requires identical source, frozen inputs and runtime")
    previous_steps = restored["steps"] if restored else 0
    if previous_steps >= config.training.total_steps:
        raise ValueError(
            "original budget exhausted; use initialize_from for a new experiment"
        )
    if config.runtime.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
    TrainingDisplay.preflight(config.logging)
    initialized = None
    if root is None:
        root, record = _allocate(config, prepared, "training")
    else:
        root = Path(root)
        record = json.loads((root / "run.json").read_text())
        record.setdefault("attempts", []).append(
            {"status": record["status"], "paths": dict(record["paths"])}
        )
        record.update(status="running")
        record.pop("failure", None)
    bind(root, config.output.name)
    try:
        write_json(root / "config/grid_recipe.json", primitive(recipe))
        attempt = (
            root
            / "evidence/training"
            / f"attempt-{len(record.get('attempts', [])):03d}"
        )
        attempt.parent.mkdir(parents=True, exist_ok=True)
        evidence = ProductionEvidence(
            attempt,
            Path(resume_from) if resume_from else None,
            resource=recipe.algorithm.provider == "rllib.resource_ppo",
            every_seconds=1.0,
        )
        record["paths"]["training"] = str(attempt.relative_to(root))
        write_json(root / "run.json", record)
    except BaseException as exc:
        exc.run_dir = root
        try:
            _finish(
                root,
                record,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                failure={"exception": type(exc).__name__, "message": str(exc)},
            )
        except Exception as metadata_error:
            exc.add_note(f"failure metadata could not be saved: {metadata_error}")
        if isinstance(exc, Exception):
            from smartsom.experiments.training import TrainingFailedError

            raise TrainingFailedError(root, exc) from exc
        raise
    state = {
        "steps": previous_steps,
        "updates": restored["updates"] if restored else 0,
        "best_score": restored["best_score"] if restored else None,
        "no_improvement": restored["no_improvement"] if restored else 0,
        "best": restored["best"] if restored else None,
        "last": str(Path(resume_from).relative_to(root)) if resume_from else None,
        "status": "completed",
        "interrupted": False,
    }
    prior_signal = None

    def interrupt(*_):
        request_interrupt(state)

    if threading.current_thread() is threading.main_thread():
        prior_signal = signal.signal(signal.SIGINT, interrupt)

    def episode(index):
        return recipe.episode(episode_root(config.seed, index))

    try:
        if initialize_from:
            from smartsom.experiments.packaging import _checkpoint_files, import_bundle
            from smartsom.learning.production_contract import (
                validate_checkpoint_manifest,
            )

            initial_source = Path(initialize_from).resolve()
            if initial_source.is_file():
                initial_source = import_bundle(
                    initial_source, root / "evidence/initialized-model"
                )
            initialized = model_locator(initial_source)
            meta = _checkpoint_files(initialized)
            validate_checkpoint_manifest(meta, recipe.scenario, recipe.algorithm)
            record["initialize_from"] = str(initialize_from)
            write_json(root / "run.json", record)
        with TrainingDisplay(root, config, on_progress=on_progress) as display:
            display.projection = evidence.presentation
            evidence.on_progress = display

            def updated(steps, updates, metrics, save):
                state.update(steps=steps, updates=updates)
                evidence.sampled_steps, evidence.updates = steps, updates
                count = steps // config.training.steps_per_update
                numeric = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (float, int)) and not isinstance(v, bool)
                }
                if any(not math.isfinite(v) for v in numeric.values()):
                    raise ValueError("nonfinite learner metric")
                evidence.append(
                    "learner_metrics.jsonl",
                    {"sampled_steps": steps, "update": updates, "metrics": numeric},
                )
                emit(str(root), {"stage": "updating"})
                update_decision = display(
                    {
                        "stage": "learning_metrics",
                        "sampled_steps": steps,
                        "learner_updates": updates,
                        "ppo_updates": count,
                        "metrics": numeric,
                    }
                )
                validate = (
                    config.validation.enabled
                    and count % config.validation.every_updates == 0
                )
                requested_stop = (
                    update_decision.get("stop")
                    if isinstance(update_decision, dict)
                    else None
                )
                stopped = state["interrupted"] or requested_stop in (
                    "pruned",
                    "interrupted",
                )
                if stopped:
                    state["status"] = (
                        "pruned" if requested_stop == "pruned" else "interrupted"
                    )
                save_last = config.checkpointing.save_last and (
                    steps == config.training.total_steps
                    or stopped
                    or config.checkpointing.every_updates is not None
                    and count % config.checkpointing.every_updates == 0
                )
                if not validate and not save_last:
                    return stopped
                checkpoint = root / "checkpoints" / f"update-{count:06d}"
                if checkpoint.exists():
                    raise ValueError("immutable update checkpoint already exists")
                selected = False
                with isolated_rng():
                    save(checkpoint)
                    write_json(
                        checkpoint / "recipe.json",
                        {"recipe": primitive(recipe), "config": primitive(config)},
                    )
                    from smartsom.trace.production import seal_checkpoint

                    emit(str(root), {"stage": "saving"})
                    seal_checkpoint(checkpoint)
                    for name in ("episodes.jsonl", "learner_metrics.jsonl"):
                        shutil.copyfile(attempt / name, checkpoint / name)
                    evidence.save_active(checkpoint)
                    if recipe.algorithm.extensions:
                        write_json(
                            checkpoint / "extension_state.json",
                            evidence.active_envs[0].extension_state["current"],
                        )
                    if prepared.validation_json:
                        write_json(
                            checkpoint / "validation_inputs.json",
                            json.loads(prepared.validation_json),
                        )
                    seal_checkpoint(checkpoint)
                    if validate:
                        report = validation_report(
                            recipe, config, checkpoint, prepared.validation_json
                        )
                        selected, reason = select_best(
                            report, state["best_score"], config.validation
                        )
                        report.update(
                            ppo_updates=count,
                            selected=selected,
                            selection_reason=reason,
                        )
                        write_json(attempt / f"validation-{count:06d}.json", report)
                        write_json(checkpoint / "validation.json", report)
                        if selected:
                            state.update(best_score=report, no_improvement=0)
                            if config.checkpointing.save_best:
                                state["best"] = str(checkpoint.relative_to(root))
                        else:
                            state["no_improvement"] += 1
                        decision = display(
                            {
                                "stage": "validation",
                                "report": report,
                                "sampled_steps": steps,
                                "ppo_updates": count,
                            }
                        )
                        if (
                            config.validation.patience is not None
                            and state["no_improvement"] >= config.validation.patience
                        ):
                            state["status"], stopped = "early_stopped", True
                        if (
                            isinstance(decision, dict)
                            and decision.get("stop") == "pruned"
                        ):
                            state["status"], stopped = "pruned", True
                    if stopped:
                        save_last = config.checkpointing.save_last
                        if state["status"] == "completed":
                            state["status"] = "interrupted"
                    if save_last:
                        state["last"] = str(checkpoint.relative_to(root))
                    if save_last or selected and config.checkpointing.save_best:
                        seal_checkpoint(checkpoint)
                        members = {
                            str(p.relative_to(checkpoint)): file_hash(p)
                            for p in sorted(checkpoint.rglob("*"))
                            if p.is_file()
                        }
                        write_json(
                            checkpoint / "update.json",
                            {**state, "identity": signature, "files": members},
                        )
                        for name in ("last", "best"):
                            if state[name]:
                                write_json(
                                    root / "checkpoints" / f"{name}.json",
                                    {
                                        "checkpoint": str(
                                            (root / state[name]).relative_to(
                                                root / "checkpoints"
                                            )
                                        )
                                    },
                                )
                    else:
                        shutil.rmtree(checkpoint)
                    retain_checkpoints(root, state, config.checkpointing.keep_last)
                return stopped

            with backend_diagnostics():
                if recipe.algorithm.provider == "sb3.maskable_ppo":
                    from smartsom.learning.production import train_sb3 as backend
                else:
                    from smartsom.learning.production_ray import train_ray as backend
                from smartsom.learning.production_sampling import ProductionSamplingSpec

                backend(
                    recipe.scenario,
                    recipe.algorithm,
                    attempt,
                    total_steps=config.training.total_steps - previous_steps,
                    rollout_steps=config.training.steps_per_update,
                    resume_from=resume_from,
                    initialize_from=initialized,
                    on_update=updated,
                    runtime=config.runtime,
                    episode_source=episode,
                    sampling_spec=ProductionSamplingSpec(
                        recipe,
                        config.seed,
                        (initialized / "extension_state.json").read_text()
                        if initialized and recipe.algorithm.extensions
                        else None,
                    ),
                    evidence=evidence,
                )
            write_json(
                attempt / "training_controls.json",
                {
                    **primitive(config.runtime),
                    **primitive(config.checkpointing),
                    "checkpoint_every_updates": config.checkpointing.every_updates,
                    "validation": primitive(config.validation)
                    if config.validation.enabled
                    else None,
                    "validation_inputs_json": prepared.validation_json,
                },
            )
            if prepared.validation_json:
                write_json(
                    attempt / "validation_inputs.json",
                    json.loads(prepared.validation_json),
                )
            evidence.progress(state["status"], force=True)
        for name in ("last", "best"):
            if state[name]:
                record["paths"][name] = state[name]
        if state["last"]:
            record["paths"]["checkpoint"] = state["last"]
        _finish(root, record, state["status"])
        evidence.save_active(attempt)
        write_json(
            attempt / "summary.json",
            {
                "status": state["status"],
                "environment_steps": state["steps"],
                "learner_updates": state["updates"],
            },
        )
        emit(
            str(root), {"stage": state["status"], "status": state["status"]}, final=True
        )
        return TrainingResult(
            root,
            attempt,
            root / state["last"] if state["last"] else None,
            root / state["best"] if state["best"] else None,
            state["steps"],
            state["updates"],
            state["steps"] // config.training.steps_per_update,
            state["status"],
        )
    except BaseException as exc:
        exc.run_dir = root
        # Keep the last coordinator-owned snapshots even when no update was
        # checkpointed. They describe partial execution, not resumable weights.
        try:
            evidence.save_active(attempt)
        except Exception as snapshot_error:
            exc.add_note(
                f"partial training evidence could not be saved: {snapshot_error}"
            )
        try:
            _finish(
                root,
                record,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                failure={"exception": type(exc).__name__, "message": str(exc)},
            )
        except Exception as metadata_error:
            exc.add_note(f"failure metadata could not be saved: {metadata_error}")
        if isinstance(exc, Exception):
            from smartsom.experiments.training import TrainingFailedError

            raise TrainingFailedError(root, exc) from exc
        raise
    finally:
        if prior_signal is not None:
            signal.signal(signal.SIGINT, prior_signal)


def resume(source, *, on_progress=None):
    source = Path(source).resolve()
    roots = [
        p
        for p in (source, *source.parents)
        if (p / "config/grid_recipe.json").is_file()
    ]
    if not roots:
        raise ValueError("resume requires a grid experiment with update checkpoints")
    root = roots[0]
    config = ExperimentConfig.model_validate_json(
        (root / "config/experiment.json").read_text()
    )
    recipe = ProductionRecipe(
        **json.loads((root / "config/grid_recipe.json").read_text())
    )
    validation = root / "config/validation_inputs.json"
    validation_json = validation.read_text() if validation.is_file() else None
    prepared = PreparedExperiment(
        canonical_json(config),
        (root / "config/origins.json").read_text(),
        recipe,
        digest(recipe_identity(recipe, config, validation_json)),
        validation_json,
    )
    checkpoint = (
        source
        if (source / "update.json").is_file()
        else root / json.loads((root / "run.json").read_text())["paths"]["checkpoint"]
    )
    return train_prepared(
        prepared, root=root, resume_from=checkpoint, on_progress=on_progress
    )
