"""Checkpoint evaluation with once-materialized worlds and retained run evidence."""

import importlib.util
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from uuid import uuid4

from smartsom.config.algorithm_binding import bind_algorithm
from smartsom.config.codec import ConfigurationError, digest, primitive, read_model
from smartsom.config.models import (
    AlgorithmFile,
    CPSatAlgorithm,
    DispatchRuleAlgorithm,
    EpisodeBudget,
    LearningAlgorithm,
    RecordingSpec,
    ResourceLearningAlgorithm,
    RunSpec,
)
from smartsom.config.resolver import SourceFile, StudySeedOrigin, _resolve_run_spec
from smartsom.config.seeds import derive_seeds
from smartsom.config.snapshots import validate_resolved
from smartsom.config.study import study_roots
from smartsom.config.training import episode_input
from smartsom.engine import DeadlockError
from smartsom.experiments.audit import audit_run
from smartsom.experiments.coverage import input_coverage
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.packaging import locate_reference
from smartsom.experiments.references import protect_model_reference
from smartsom.experiments.runner import RunFailedError, run_one
from smartsom.experiments.training_audit import load_training_snapshot
from smartsom.learning.checkpoint import (
    CheckpointManifest,
    ResourceCheckpointManifest,
    file_hash,
)
from smartsom.learning.joint import PolicyStalledError


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    run_dir: Path
    status: str
    completed: int
    failed: int
    engineering_failures: int
    results: tuple[dict, ...]
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class _Options:
    seed: int = 202
    replications: int = 5
    deterministic: bool = True
    full_replay: bool = True
    checkpoint: str = "last"
    baselines: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()

    @classmethod
    def freeze(cls, options):
        row = cls(
            **{
                key: getattr(options, key, value.default)
                for key, value in cls.__dataclass_fields__.items()
            }
        )
        if type(row.seed) is not int or not 0 <= row.seed < 2**64:
            raise ConfigurationError(
                "evaluation seed must be an unsigned 64-bit integer"
            )
        if type(row.replications) is not int or row.replications < 1:
            raise ConfigurationError("evaluation replications must be positive")
        if type(row.deterministic) is not bool or type(row.full_replay) is not bool:
            raise ConfigurationError(
                "evaluation deterministic/full_replay must be booleans"
            )
        if row.checkpoint not in ("last", "best"):
            raise ConfigurationError("evaluation checkpoint must be last or best")
        for key in ("baselines", "scenarios"):
            values = getattr(row, key)
            if not isinstance(values, (tuple, list)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ConfigurationError(
                    f"evaluation {key} must be a sequence of nonempty strings"
                )
            row = replace(row, **{key: tuple(values)})
        return row


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    directory: Path
    metadata: CheckpointManifest
    sha256: str
    snapshot: Path | None
    selection: str

    def algorithm(self):
        kind = (
            ResourceLearningAlgorithm
            if self.metadata.provider == "rllib.resource_ppo"
            else LearningAlgorithm
        )
        return AlgorithmFile(
            schema="smartsom.algorithm/v1",
            algorithm=kind(
                provider=self.metadata.provider,
                projection=self.metadata.projection,
                parameters=self.metadata.parameters,
                checkpoint=str(self.directory),
                checkpoint_sha256=self.sha256,
            ),
        )

    def identity(self):
        return {
            "path": str(self.directory),
            "manifest_sha256": self.sha256,
            "provider": self.metadata.provider,
            "selection": self.selection,
            "final_weights_sha256": self.metadata.final_weights_sha256,
            "training_snapshot": str(self.snapshot) if self.snapshot else None,
            "training_snapshot_sha256": file_hash(self.snapshot)
            if self.snapshot
            else None,
        }


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return value


def _reference(owner, value):
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"invalid checkpoint reference in {owner}")
    try:
        return locate_reference(owner, value)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"invalid reference in {owner}: {exc}") from exc


def _select_checkpoint(source, selection="last"):
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.name in ("run.json", "checkpoint.json"):
        path = path.parent
    original = path
    candidates = []
    outer = path / "run.json"
    if outer.exists():
        record = _json(outer)
        if record.get("schema") == "smartsom.evaluation/v1":
            checkpoint = record.get("checkpoint", {})
            path = _reference(outer, checkpoint.get("path"))
            if checkpoint.get("training_snapshot"):
                candidates.append(_reference(outer, checkpoint["training_snapshot"]))
        training = record.get("paths", {}).get("training")
        if training:
            path = _reference(outer, training)
            candidates.append(path / "resolved_training.json")
        alias = original / "checkpoints" / selection
        if alias.exists():
            path = alias.resolve()
        elif not training and path == original:
            raise ConfigurationError(
                "experiment has no saved training/checkpoint selection"
            )
    explicit = (path / "checkpoint.json").is_file() or (
        path / "inference/checkpoint.json"
    ).is_file()
    if not explicit:
        pointer = path / "checkpoints" / f"{selection}.json"
        if pointer.exists():
            candidates.append(path / "resolved_training.json")
            path = _reference(pointer, _json(pointer).get("checkpoint"))
        elif selection == "last" and (path / "checkpoint/checkpoint.json").is_file():
            candidates.append(path / "resolved_training.json")
            path /= "checkpoint"
        else:
            raise ConfigurationError(f"no {selection} checkpoint available in {path}")
    candidates = [
        path / "resolved_training.json",
        path.parent / "resolved_training.json",
        *candidates,
    ]
    if (path / "inference/checkpoint.json").is_file():
        path /= "inference"
    metadata_path = path / "checkpoint.json"
    provider = _json(metadata_path).get("provider")
    kind = (
        ResourceCheckpointManifest
        if provider == "rllib.resource_ppo"
        else CheckpointManifest
    )
    metadata, sha = read_model(metadata_path, kind)
    snapshot = next(
        (candidate for candidate in candidates if candidate.is_file()), None
    )
    return _Checkpoint(
        path, metadata, sha, snapshot, "explicit" if explicit else selection
    )


def _algorithms(selected, names):
    result = [("model", selected.algorithm(), selected)]
    aliases = {
        "spt": "builtin.spt",
        "builtin.spt": "builtin.spt",
        "first_feasible": "builtin.first_feasible",
        "builtin.first_feasible": "builtin.first_feasible",
        "cp": "pyjobshop.cp_sat",
        "cp_sat": "pyjobshop.cp_sat",
        "pyjobshop.cp_sat": "pyjobshop.cp_sat",
    }
    for name in names:
        provider = aliases.get(name)
        if provider:
            cp = provider == "pyjobshop.cp_sat"
            spec = (
                CPSatAlgorithm(
                    provider=provider,
                    interface_kind="offline_solver",
                    required_information="full_static",
                )
                if cp
                else DispatchRuleAlgorithm(provider=provider)
            )
            result.append(
                (
                    provider,
                    AlgorithmFile(schema="smartsom.algorithm/v1", algorithm=spec),
                    None,
                )
            )
        else:
            other = _select_checkpoint(name.removeprefix("checkpoint:"))
            result.append((f"checkpoint-{other.sha256[:16]}", other.algorithm(), other))
    if len({row[0] for row in result}) != len(result):
        raise ConfigurationError("evaluation contains duplicate baselines")
    return result


def _training_world(training, snapshot, world, replication):
    episode = training.episode(replication, root_seed=world)
    inp = episode.input
    base = training.base
    updates = {
        name: getattr(inp, name)
        for name in ("arrivals", "processing_times", "machine_events", "quality")
    }
    for name in ("arrivals", "processing_times", "machine_events"):
        value = updates[name]
        updates[f"{name}_sha256"] = digest(value) if value is not None else None
    updates["quality_draws_sha256"] = digest(inp.quality.draws) if inp.quality else None
    updates["quality_modes_sha256"] = digest(inp.quality.modes) if inp.quality else None
    updates.update(
        {
            name: getattr(episode, name)
            for name in (
                "arrival_provenance",
                "processing_provenance",
                "machine_event_provenance",
                "quality_provenance",
            )
        }
    )
    return replace(
        base,
        **updates,
        seeds=episode.seeds,
        sources=(SourceFile("training_snapshot", snapshot, file_hash(snapshot)),),
        run=base.run.model_copy(update={"seed": world, "budget": None}),
    )


def _case_world(path, world):
    return _resolve_run_spec(
        RunSpec(
            schema="smartsom.run/v1",
            scenario=str(path),
            algorithm="__evaluation_world__",
            seed=world,
            output_root="runs",
        ),
        path,
        [],
        algorithm_override=AlgorithmFile(
            schema="smartsom.algorithm/v1",
            algorithm=DispatchRuleAlgorithm(provider="builtin.spt"),
        ),
    )


def _bind(base, algorithm, algorithm_id, case_id, replication, options, limits):
    world, algorithm_root = study_roots(
        options.seed, case_id, replication, algorithm_id
    )
    cp = isinstance(algorithm.algorithm, CPSatAlgorithm)
    learning = isinstance(algorithm.algorithm, LearningAlgorithm)
    scenario = (
        base.scenario.model_copy(update={"visibility": "full_static"})
        if cp
        else base.scenario
    )
    run = bind_algorithm(
        base.run.model_copy(
            update={
                "seed": world,
                "algorithm": algorithm_id,
                "budget": limits if learning else None,
                "recording": RecordingSpec(observations="hash"),
            }
        ),
        scenario,
        algorithm,
    )
    if cp and any(
        importlib.util.find_spec(name) is None for name in ("pyjobshop", "ortools")
    ):
        raise ConfigurationError("CP baseline requires uv sync --locked --extra cp")
    algorithm_seeds = {
        seed.domain: seed
        for seed in derive_seeds(algorithm_root, generated=False, solver=cp)
    }
    if learning and not options.deterministic:
        algorithm_seeds["algorithm"] = replace(
            algorithm_seeds["algorithm"], consumed=True
        )
    seeds = tuple(
        algorithm_seeds[seed.domain] if seed.domain in ("algorithm", "solver") else seed
        for seed in base.seeds
    )
    return validate_resolved(
        replace(
            base,
            scenario=scenario,
            algorithm=algorithm,
            run=run,
            seeds=seeds,
            study_seed_origin=StudySeedOrigin(
                options.seed, case_id, replication, algorithm_id, world, algorithm_root
            ),
        )
    )


def _prepare(selected, options):
    training = load_training_snapshot(selected.snapshot) if selected.snapshot else None
    if not options.scenarios and training is None:
        raise ConfigurationError(
            "checkpoint has no training snapshot; supply evaluation scenarios"
        )
    limits = training.run.budget.limits() if training else None
    budget = (
        EpisodeBudget(max_decisions=limits.max_decisions, max_ticks=limits.max_ticks)
        if limits
        else EpisodeBudget()
    )
    cases = []
    for reference in options.scenarios:
        path = Path(reference).expanduser().resolve()
        if path.is_dir():
            path /= "scenario.yaml"
        try:
            sha = file_hash(path)
        except OSError as exc:
            raise ConfigurationError(
                f"cannot read evaluation scenario {path}: {exc}"
            ) from exc
        cases.append((f"{path.stem}-{sha[:12]}", path))
    if not cases:
        cases = [("training", None)]
    if len({case[0] for case in cases}) != len(cases):
        raise ConfigurationError("evaluation contains duplicate scenario identities")
    algorithms = _algorithms(selected, options.baselines)
    entries = []
    for case_id, path in cases:
        for replication in range(options.replications):
            world, _ = study_roots(options.seed, case_id, replication, "")
            base = (
                _case_world(path, world)
                if path
                else _training_world(training, selected.snapshot, world, replication)
            )
            world_sha = digest(episode_input(base))
            for algorithm_id, algorithm, checkpoint in algorithms:
                resolved = _bind(
                    base, algorithm, algorithm_id, case_id, replication, options, budget
                )
                entries.append(
                    (
                        resolved,
                        {
                            "case_id": case_id,
                            "replication": replication,
                            "algorithm_id": algorithm_id,
                            "provider": algorithm.algorithm.provider,
                            "world_sha256": world_sha,
                            "world_seed": world,
                            "input_sha256": {
                                name: getattr(resolved, f"{name}_sha256")
                                for name in (
                                    "factory",
                                    "workload",
                                    "arrivals",
                                    "processing_times",
                                    "machine_events",
                                    "quality_draws",
                                    "quality_modes",
                                )
                            },
                            "seeds": primitive(resolved.seeds),
                            "checkpoint": checkpoint.identity() if checkpoint else None,
                            "information": algorithm.algorithm.required_information,
                            "scenario_visibility": resolved.scenario.visibility,
                        },
                    )
                )
    return entries, [
        {
            "case_id": case_id,
            "scenario": str(path) if path else None,
            "training_snapshot": str(selected.snapshot) if path is None else None,
        }
        for case_id, path in cases
    ]


def _failure(exc):
    if isinstance(exc, PolicyStalledError) or str(exc).startswith("policy_stalled:"):
        return "policy_stalled", False
    if isinstance(exc, DeadlockError):
        return "deadlock", False
    if str(exc).startswith("budget_exhausted:"):
        return "budget_exhausted", False
    if str(exc).startswith("solver returned no feasible incumbent"):
        return "solver_no_incumbent", False
    return type(exc).__name__, True


def _summary(rows, requested):
    completed = sum(row["status"] == "completed" for row in rows)
    failed = len(rows) - completed
    engineering = sum(row.get("engineering_failure", False) for row in rows)
    aggregates = []
    for case_id, algorithm_id in sorted(
        {(row["case_id"], row["algorithm_id"]) for row in rows}
    ):
        group = [
            row
            for row in rows
            if (row["case_id"], row["algorithm_id"]) == (case_id, algorithm_id)
        ]
        values = [row["makespan"] for row in group if row["status"] == "completed"]
        aggregates.append(
            {
                "case_id": case_id,
                "algorithm_id": algorithm_id,
                "completed": len(values),
                "failed": len(group) - len(values),
                "makespan_mean": mean(values) if values else None,
                "makespan_sample_std": stdev(values) if len(values) > 1 else None,
            }
        )
    return {
        "completed": completed,
        "failed": failed,
        "engineering_failures": engineering,
        "pending": requested - len(rows),
        "aggregates": aggregates,
    }


def evaluate_checkpoint(source, options, *, output_root=None) -> EvaluationResult:
    """Evaluate the selected export and explicit baselines; never train or overwrite.

    Legacy training/checkpoint directories and outer experiment run.json files are
    accepted. Model-only evaluation is the default. Saved training input snapshots
    supply the default case, without reopening historical authoring references.
    """
    options = _Options.freeze(options)
    selected = _select_checkpoint(source, options.checkpoint)
    entries, cases = _prepare(selected, options)
    coverage = input_coverage(selected.snapshot, entries)
    root = Path(output_root or "runs").expanduser().resolve()
    directory = root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-evaluation-" + uuid4().hex[:10]
    )
    directory.mkdir(parents=True)
    record = {
        "schema": "smartsom.evaluation/v1",
        "id": directory.name,
        "kind": "evaluation",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "source": source_identity(),
        "checkpoint": selected.identity(),
        "options": primitive(options),
        "cases": cases,
        "input_coverage": coverage,
        "requested": len(entries),
        "results": [],
        "paths": {
            "summary": "summary.json",
            "runs": "evidence/runs",
            "plan": "plan.json",
        },
    }
    write_json(directory / "plan.json", {"entries": [row for _, row in entries]})
    for model in {row["checkpoint"]["path"] for _, row in entries if row["checkpoint"]}:
        protect_model_reference(model, directory / "run.json")
    record.update(_summary([], len(entries)))
    write_json(directory / "run.json", record)
    write_json(
        directory / "summary.json",
        {"status": "running", "input_coverage": coverage, **_summary([], len(entries))},
    )
    try:
        for resolved, planned in entries:
            row = {
                **planned,
                "run_dir": None,
                "makespan": None,
                "engineering_failure": False,
            }
            resolved = replace(
                resolved,
                run=resolved.run.model_copy(
                    update={"output_root": str(directory / "evidence/runs")}
                ),
            )
            try:
                result = run_one(resolved, deterministic=options.deterministic)
                row.update(
                    status="completed",
                    run_dir=str(result.run_dir.relative_to(directory)),
                    makespan=result.simulation_result.makespan,
                )
            except RunFailedError as exc:
                if isinstance(exc.cause, KeyboardInterrupt):
                    row.update(
                        status="interrupted",
                        reason="KeyboardInterrupt",
                        run_dir=str(exc.run_dir.relative_to(directory)),
                        replay={"status": "not_run_interrupted"},
                    )
                    record["results"].append(row)
                    raise exc.cause
                reason, engineering = _failure(exc.cause)
                row.update(
                    status="failed" if engineering else "not_completed",
                    reason=reason,
                    engineering_failure=engineering,
                    run_dir=str(exc.run_dir.relative_to(directory)),
                    error=str(exc.cause),
                )
            except Exception as exc:
                reason, _ = _failure(exc)
                row.update(
                    status="failed",
                    reason=reason,
                    engineering_failure=True,
                    error=str(exc),
                )
            if options.full_replay and row["run_dir"]:
                try:
                    row["replay"] = audit_run(directory / row["run_dir"])
                except Exception as exc:
                    row.update(
                        status="failed",
                        makespan=None,
                        engineering_failure=True,
                        replay={"status": "failed", "error": str(exc)},
                    )
            else:
                row["replay"] = {
                    "status": "not_requested"
                    if not options.full_replay
                    else "unavailable_no_run_evidence"
                }
            record["results"].append(row)
            summary = _summary(record["results"], len(entries))
            record.update(summary)
            write_json(
                directory / "summary.json", {"input_coverage": coverage, **summary}
            )
            write_json(directory / "run.json", record)
        record["status"] = (
            "failed"
            if record["engineering_failures"]
            else "completed_with_failures"
            if record["failed"]
            else "completed"
        )
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        record.update(_summary(record["results"], len(entries)))
        record["finished_at"] = datetime.now(UTC).isoformat()
        write_json(directory / "run.json", record)
        write_json(
            directory / "summary.json",
            {
                "status": record["status"],
                "input_coverage": coverage,
                **_summary(record["results"], len(entries)),
            },
        )
        raise
    record["finished_at"] = datetime.now(UTC).isoformat()
    write_json(directory / "run.json", record)
    write_json(
        directory / "summary.json",
        {
            "status": record["status"],
            "input_coverage": coverage,
            **_summary(record["results"], len(entries)),
        },
    )
    return EvaluationResult(
        directory,
        record["status"],
        record["completed"],
        record["failed"],
        record["engineering_failures"],
        tuple(record["results"]),
        selected.directory,
    )
