"""Bounded local training studies, separate from the v1 simulation study contract."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import signal
import time
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, canonical_json, digest, primitive
from smartsom.config.experiment import (
    ExperimentConfig,
    PreparedExperiment,
    prepare,
    prepare_frozen,
    training_identity,
)
from smartsom.experiments.batch import exclusive_lock, execution_identity
from smartsom.experiments.catalog import contained_path, read_json
from smartsom.experiments.evidence import _file_digest, write_json
from smartsom.experiments.search import (
    OptunaSession,
    candidates,
    final_validation,
    require_optuna,
    trial_configs,
    validate_search,
    validation_score,
)
from smartsom.telemetry.runtime import (
    CURRENT,
    backend_diagnostics,
    bind,
    emit,
    operation,
    worker_output,
)

PLAN_SCHEMA = "smartsom.learning-study-plan/v1"
TRIAL_SCHEMA = "smartsom.learning-trial/v1"
SUCCESS = {"completed", "early_stopped"}


@dataclass(frozen=True)
class LearningStudyResult:
    run_dir: Path
    status: str
    completed: int
    failed: int
    pending: int
    entries: tuple[dict, ...]


def _atomic(path, value):
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _freeze(config):
    return _freeze_prepared(prepare(config))


def _freeze_prepared(prepared):
    return {
        "config": json.loads(prepared.config_json),
        "config_sha256": digest(json.loads(prepared.config_json)),
        "origins": json.loads(prepared.origins_json),
        "resolved_training": primitive(prepared.resolved),
        "scientific_sha256": prepared.scientific_sha256,
        "validation_json": prepared.validation_json,
    }


def _trial(configs, index, parameters=None, *, templates=None):
    rows = [
        _freeze(config)
        if templates is None
        else _freeze_prepared(prepare_frozen(config, templates[config.seed]))
        for config in configs
    ]
    return {
        "schema": TRIAL_SCHEMA,
        "index": index,
        "id": f"trial-{index:06d}-{digest(rows)[:10]}",
        "parameters": parameters or {},
        "configs": rows,
    }


def _templates(plan):
    """Restore seed-specific materialized worlds without reading authoring paths."""
    from smartsom.experiments.training_audit import TrainingSnapshot

    rows = plan.get("templates")
    if not rows:
        raise ConfigurationError(
            "Optuna plan has no frozen templates; create a new search"
        )
    templates = {}
    for frozen in rows:
        config = ExperimentConfig.model_validate_json(canonical_json(frozen["config"]))
        if digest(frozen["config"]) != frozen["config_sha256"]:
            raise ConfigurationError("frozen template configuration digest mismatch")
        snapshot = TrainingSnapshot.model_validate_json(
            canonical_json(
                {
                    "schema": "smartsom.resolved-training/v1",
                    "resolved": frozen["resolved_training"],
                }
            )
        )
        prepared = PreparedExperiment(
            canonical_json(frozen["config"]),
            canonical_json(frozen["origins"]),
            snapshot.resolved,
            frozen["scientific_sha256"],
            frozen.get("validation_json"),
        )
        # Verify the frozen scientific identity before consuming a proposal slot.
        prepare_frozen(config, prepared)
        if config.seed in templates:
            raise ConfigurationError("duplicate frozen template seed")
        templates[config.seed] = prepared
    if set(templates) != set(plan["base"]["search"]["seeds"]):
        raise ConfigurationError(
            "frozen templates do not cover the declared training seeds"
        )
    return templates


def _allocate(plan, output_root, name):
    identity = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{name}-{uuid4().hex[:10]}"
    )
    root = Path(output_root).resolve() / identity
    root.mkdir(parents=True)
    for directory in ("config", "trials", "logs"):
        (root / directory).mkdir()
    write_json(root / "config/plan.json", plan)
    record = {
        "schema": "smartsom.experiment/v2",
        "id": identity,
        "name": name,
        "kind": plan["kind"],
        "status": "prepared",
        "paths": {
            "config": "config/plan.json",
            "logs": "logs",
            "report": "summary.json",
        },
        "execution_identity": execution_identity(),
        "plan_sha256": _file_digest(root / "config/plan.json"),
        "trials": [],
    }
    _atomic(root / "run.json", record)
    for trial in plan.get("trials", []):
        _save_trial(root, record, trial)
    return root, record


def _save_trial(root, record, trial):
    directory = root / "trials" / trial["id"]
    # ZIP inventories contain real files, so an unstarted import may have no trials/.
    directory.mkdir(parents=True)
    for replication, frozen in enumerate(trial["configs"]):
        snapshot = directory / f"snapshot-{replication:03d}.json"
        write_json(
            snapshot,
            {
                "schema": "smartsom.resolved-training/v1",
                "resolved": frozen["resolved_training"],
            },
        )
        frozen["snapshot"] = snapshot.name
        frozen["snapshot_sha256"] = _file_digest(snapshot)
    write_json(directory / "plan.json", trial)
    record["trials"].append(
        {"id": trial["id"], "plan_sha256": _file_digest(directory / "plan.json")}
    )
    _atomic(root / "run.json", record)
    return directory


def _load(root):
    record = read_json(root / "run.json")
    if record.get("schema") != "smartsom.experiment/v2" or record.get("kind") not in {
        "learning_batch",
        "search",
    }:
        raise ConfigurationError("resume requires a learning_batch/search run")
    if record["execution_identity"] != execution_identity():
        raise ConfigurationError(
            "source or dependency identity changed; create a new learning study"
        )
    plan_path = contained_path(root, record["paths"]["config"])
    if _file_digest(plan_path) != record["plan_sha256"]:
        raise ConfigurationError("learning study plan digest mismatch")
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ConfigurationError("unknown learning study plan")
    for item in record["trials"]:
        path = contained_path(root, f"trials/{item['id']}/plan.json")
        if _file_digest(path) != item["plan_sha256"]:
            raise ConfigurationError("frozen trial plan digest mismatch")
        trial = read_json(path)
        for frozen in trial["configs"]:
            if digest(frozen["config"]) != frozen["config_sha256"]:
                raise ConfigurationError(
                    "frozen training configuration digest mismatch"
                )
            snapshot = contained_path(path.parent, frozen["snapshot"])
            if _file_digest(snapshot) != frozen["snapshot_sha256"]:
                raise ConfigurationError("frozen training snapshot digest mismatch")
    return record, plan


def _evidence_hashes(run_dir):
    result = {}
    for section in ("config", "evidence", "checkpoints"):
        for directory, names, files in os.walk(run_dir / section, followlinks=False):
            names[:] = [
                name
                for name in names
                if name != "references" and not (Path(directory) / name).is_symlink()
            ]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                result[str(path.relative_to(run_dir))] = _file_digest(path)
    result["run.json"] = _file_digest(run_dir / "run.json")
    return result


def _attempts(directory):
    return sorted(directory.glob("attempt-*"))


def _state(directory):
    attempts = _attempts(directory)
    if not attempts:
        return {"status": "pending", "attempt": None}
    latest = attempts[-1]
    result = latest / "result.json"
    if not result.exists():
        return {"status": "interrupted", "attempt": str(latest)}
    data = read_json(result)
    if data["status"] in {"completed", "ineligible"}:
        configs = read_json(directory / "plan.json")["configs"]
        if len(data["runs"]) != len(configs) or {
            row["replication"] for row in data["runs"]
        } != set(range(len(configs))):
            raise ConfigurationError("completed trial replication coverage mismatch")
        for row in data["runs"]:
            if (
                row["status"] not in SUCCESS
                or row["seed"] != configs[row["replication"]]["config"]["seed"]
            ):
                raise ConfigurationError("completed trial seed/status mismatch")
            root = contained_path(directory, row["run_dir"])
            if _evidence_hashes(root) != row["artifacts"]:
                raise ConfigurationError(
                    "completed training evidence changed; refusing reuse"
                )
    return data | {"attempt": str(latest)}


def _saved_runs(attempt):
    return (
        {
            row["replication"]: row
            for row in read_json(attempt / "progress.json").get("runs", [])
        }
        if (attempt / "progress.json").exists()
        else {}
    )


def _can_resume(attempt):
    from smartsom.experiments.packaging import model_locator

    for path in attempt.glob("replication-*/*/run.json"):
        record = read_json(path)
        if record["status"] in SUCCESS:
            continue
        try:
            selected = model_locator(path.parent)
            if (selected / "update.json").is_file():
                from smartsom.experiments.production_training import verify_checkpoint

                verify_checkpoint(selected)
                continue
            if (
                selected.name != "inference"
                or not (selected.parent / "resolved_training.json").is_file()
            ):
                return False
        except (ValueError, OSError):
            return False
    return True


def _completed_result(child):
    from smartsom.experiments.packaging import model_locator

    record = read_json(child / "run.json")
    training = contained_path(child, record["paths"]["training"])
    summary = read_json(training / "summary.json")
    return SimpleNamespace(
        run_dir=child,
        training_dir=training,
        last_checkpoint=model_locator(child),
        environment_steps=summary["environment_steps"],
        learner_updates=summary["learner_updates"],
        status=record["status"],
    )


def _run_trial(
    directory,
    attempt,
    trial,
    identity,
    search_options=None,
    optuna_identity=None,
    messages=None,
):
    """One process owns each trial; training seeds execute in declared order."""
    from smartsom import api
    from smartsom.experiments.training_audit import (
        audit_training,
        load_training_snapshot,
    )

    if execution_identity() != identity:
        raise ConfigurationError("source changed before training worker execution")
    runs, reports = [], {}
    saved = _saved_runs(attempt)
    session = (
        OptunaSession(directory.parents[1], search_options) if optuna_identity else None
    )
    for replication, frozen in enumerate(trial["configs"]):
        config = ExperimentConfig.model_validate_json(canonical_json(frozen["config"]))
        snapshot = contained_path(directory, frozen["snapshot"])
        if _file_digest(snapshot) != frozen["snapshot_sha256"]:
            raise ConfigurationError("frozen training snapshot digest mismatch")
        resolved = load_training_snapshot(snapshot)
        if (
            digest(
                training_identity(
                    resolved, config.runtime, frozen.get("validation_json")
                )
            )
            != frozen["scientific_sha256"]
        ):
            raise ConfigurationError("training inputs differ from the frozen trial")
        if replication in saved and saved[replication]["status"] in SUCCESS:
            previous = saved[replication]
            child = contained_path(directory, previous["run_dir"])
            if _evidence_hashes(child) != previous["artifacts"]:
                raise ConfigurationError("completed replication evidence changed")
            runs.append(previous)
            for path in contained_path(directory, previous["training_dir"]).glob(
                "validation-*.json"
            ):
                report = read_json(path)
                reports[replication, report["ppo_updates"]] = report
            continue
        allocation = attempt / f"replication-{replication:03d}"
        config.output.root = str(allocation)
        config.output.name = f"{trial['id']}-seed{config.seed}"

        def progress(value):
            display = CURRENT.get()
            projected = (
                display.tasks.get(str(display.root), {}).get("values", {})
                if display
                else {}
            )
            if messages is not None:
                messages.put(
                    (
                        trial["id"],
                        {
                            **primitive(value),
                            "display_total": config.training.total_steps,
                            "display_unit": "sampling decisions",
                            "display_run": str(allocation),
                            "display_name": f"{trial['id']} / seed {config.seed}",
                            "display_values": dict(projected),
                        },
                    )
                )
            if (
                not search_options
                or not isinstance(value, dict)
                or value.get("stage") != "validation"
            ):
                return None
            step, report = value["ppo_updates"], value["report"]
            reports[replication, step] = report
            matching = [reports.get((r, step)) for r in range(len(trial["configs"]))]
            if session and all(row is not None for row in matching):
                values = [
                    validation_score(row, search_options.objective) for row in matching
                ]
                if all(item is not None for item in values) and session.report(
                    optuna_identity, mean(values), step
                ):
                    return {"stop": "pruned"}
            return None

        existing = sorted(path.parent for path in allocation.glob("*/run.json"))
        if len(existing) > 1:
            raise ConfigurationError("ambiguous child training run in an attempt")
        if existing and read_json(existing[0] / "run.json")["status"] in SUCCESS:
            result = _completed_result(existing[0])
        elif existing:
            result = api.resume(existing[0], on_progress=progress)
        else:
            prepared = PreparedExperiment(
                canonical_json(config),
                canonical_json(frozen["origins"]),
                resolved,
                frozen["scientific_sha256"],
                frozen.get("validation_json"),
            )
            result = api.train_prepared(prepared, on_progress=progress)
        child = Path(result.run_dir).resolve()
        row = {
            "replication": replication,
            "seed": config.seed,
            "status": result.status,
            "run_dir": str(child.relative_to(directory)),
            "training_dir": str(Path(result.training_dir).relative_to(directory)),
            "environment_steps": result.environment_steps,
            "learner_updates": result.learner_updates,
        }
        if result.status in SUCCESS:
            row["artifacts"] = _evidence_hashes(child)
            if search_options:
                row["training_audit"] = audit_training(Path(result.training_dir))
                if row["training_audit"].get("status") != "passed":
                    raise ValueError("training audit did not pass")
                updates = result.environment_steps // config.training.steps_per_update
                row["objective"] = final_validation(
                    Path(result.training_dir), updates, search_options.objective
                )
                from smartsom.experiments.packaging import model_locator

                selected = model_locator(result.last_checkpoint)
                checkpoint = read_json(selected / "checkpoint.json")
                weights = row["objective"]["weights"]
                if not weights:
                    raise ValueError(
                        "final validation has no checkpoint weight identity"
                    )
                expected = (
                    digest(weights)
                    if checkpoint["provider"] == "rllib.resource_ppo"
                    else next(iter(weights.values()))
                )
                if (
                    checkpoint["final_weights_sha256"] != expected
                    or checkpoint["environment_steps"] != result.environment_steps
                ):
                    raise ValueError(
                        "final validation and checkpoint identities differ"
                    )
                row["checkpoint"] = str(selected.relative_to(directory))
                row["checkpoint_sha256"] = _file_digest(selected / "checkpoint.json")
        runs.append(row)
        _atomic(attempt / "progress.json", {"runs": runs})
        if result.status not in SUCCESS:
            return {"status": result.status, "runs": runs, "value": None}
    values = [row["objective"]["value"] for row in runs] if search_options else []
    if execution_identity() != identity:
        raise ConfigurationError("source changed during training study execution")
    eligible = not search_options or all(value is not None for value in values)
    return {
        "status": "completed" if eligible else "ineligible",
        "runs": runs,
        "value": mean(values) if values and eligible else None,
    }


@worker_output("attempt")
def _worker(
    directory, attempt, trial, identity, search_data, optuna_identity, messages
):
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    directory, attempt = Path(directory), Path(attempt)
    try:
        from smartsom.config.experiment import SearchOptions

        options = (
            SearchOptions.model_validate_json(canonical_json(search_data))
            if search_data
            else None
        )
        with exclusive_lock(directory / "worker.lock"):
            result = _run_trial(
                directory, attempt, trial, identity, options, optuna_identity, messages
            )
    except BaseException as exc:
        result = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "runs": list(_saved_runs(attempt).values()),
            "value": None,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
    _atomic(attempt / "result.json", result)
    messages.put((trial["id"], {"stage": "worker_exit"}))


def _summary(root, record, *, interrupted=False):
    entries = []
    for item in record["trials"]:
        directory = root / "trials" / item["id"]
        state = _state(directory)
        entries.append({"id": item["id"], **state})
    counts = Counter(row["status"] for row in entries)
    completed = counts["completed"]
    failed = counts["failed"] + counts["ineligible"]
    pending = record["trial_budget"] - sum(
        counts[state] for state in ("completed", "failed", "ineligible", "pruned")
    )
    status = (
        "interrupted"
        if interrupted or pending
        else "completed"
        if not failed
        else "finished_with_failures"
    )
    record["status"] = status
    _atomic(root / "run.json", record)
    summary = {
        "schema": "smartsom.learning-study-summary/v1",
        "status": status,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "pruned": counts["pruned"],
        "entries": entries,
    }
    scored = [
        entry
        for entry in entries
        if entry["status"] == "completed" and entry.get("value") is not None
    ]
    if scored:
        summary["best_trial"] = sorted(
            scored,
            key=lambda row: (
                row["value"] if record.get("direction") != "max" else -row["value"],
                row["id"],
            ),
        )[0]["id"]
    _atomic(root / "summary.json", summary)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "status", "value", "attempt"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(entries)
    return LearningStudyResult(root, status, completed, failed, pending, tuple(entries))


@operation("batch-train")
def _execute(root, *, retry_failed=False, on_progress=None):
    context = multiprocessing.get_context("spawn")
    root = Path(root).resolve()
    with exclusive_lock(root / "study.lock"), ExitStack() as diagnostics:
        record, plan = _load(root)
        previous_tasks = {}
        if (root / "logs/progress.json").is_file():
            from smartsom.telemetry.monitor import read_snapshot

            try:
                previous_tasks = {
                    row["id"]: row for row in read_snapshot(root)["tasks"]
                }
            except (OSError, ValueError):
                pass
        display = bind(root)
        # Retain recorded counters when a completed trial is not executed again.
        # The authoritative trial state below always replaces snapshot status.
        display.tasks.update(
            (item["id"], previous_tasks[item["id"]])
            for item in record["trials"]
            if item["id"] in previous_tasks
        )
        diagnostics.enter_context(backend_diagnostics(display))
        display.total_tasks = record["trial_budget"]
        base = (
            ExperimentConfig.model_validate_json(canonical_json(plan["base"]))
            if plan.get("base")
            else None
        )
        options = base.search if record["kind"] == "search" else None
        templates = _templates(plan) if options and options.method == "optuna" else None
        session = (
            OptunaSession(root, options)
            if options and options.method == "optuna"
            else None
        )
        queue = deque()
        for item in record["trials"]:
            directory = root / "trials" / item["id"]
            state = _state(directory)
            emit(
                item["id"],
                {"stage": state["status"], "status": state["status"]},
                final=True,
            )
            if session and state["attempt"] is not None:
                attempt_path = Path(state["attempt"]) / "attempt.json"
                if attempt_path.exists():
                    attempt_record = read_json(attempt_path)
                    session.finish(
                        attempt_record["optuna"], state["status"], state.get("value")
                    )
            if (
                state["status"] in {"pending", "interrupted"}
                or retry_failed
                and state["status"] in {"failed", "ineligible"}
                and read_json(directory / "plan.json")["configs"]
            ):
                queue.append(directory)
        active, messages, interrupted = {}, context.Queue(), False
        record["status"] = "running"
        _atomic(root / "run.json", record)
        try:
            while (
                queue
                or active
                or session
                and len(record["trials"]) < record["trial_budget"]
            ):
                try:
                    while not interrupted and len(active) < plan["max_concurrent"]:
                        if not queue:
                            if (
                                not session
                                or len(record["trials"]) >= record["trial_budget"]
                            ):
                                break
                            parameters, optuna_identity = session.ask(
                                len(record["trials"])
                            )
                            try:
                                trial = _trial(
                                    trial_configs(base, parameters),
                                    len(record["trials"]),
                                    parameters,
                                    templates=templates,
                                )
                            except Exception as exc:
                                session.finish(optuna_identity, "failed", None)
                                trial = {
                                    "schema": TRIAL_SCHEMA,
                                    "id": f"trial-{len(record['trials']):06d}-invalid",
                                    "index": len(record["trials"]),
                                    "parameters": parameters,
                                    "configs": [],
                                }
                                directory = _save_trial(root, record, trial)
                                attempt = directory / "attempt-000"
                                attempt.mkdir()
                                _atomic(
                                    attempt / "result.json",
                                    {
                                        "status": "failed",
                                        "runs": [],
                                        "value": None,
                                        "failure": {
                                            "type": type(exc).__name__,
                                            "message": str(exc),
                                        },
                                    },
                                )
                                continue
                            trial["optuna"] = optuna_identity
                            queue.append(_save_trial(root, record, trial))
                        directory = queue.popleft()
                        trial = read_json(directory / "plan.json")
                        state = _state(directory)
                        previous = _attempts(directory)
                        if (
                            state["status"] == "interrupted"
                            and previous
                            and _can_resume(previous[-1])
                        ):
                            attempt = previous[-1]
                            attempt_record = read_json(attempt / "attempt.json")
                        else:
                            attempt = directory / f"attempt-{len(previous):03d}"
                            attempt.mkdir()
                            attempt_record = {"schema": "smartsom.training-attempt/v1"}
                            if state["status"] == "interrupted" and previous:
                                old = read_json(previous[-1] / "attempt.json")
                                attempt_record["previous_attempt"] = previous[-1].name
                                preserved = [
                                    row
                                    for row in _saved_runs(previous[-1]).values()
                                    if row["status"] in SUCCESS
                                ]
                                _atomic(attempt / "progress.json", {"runs": preserved})
                            if session:
                                attempt_record["optuna"] = (
                                    old["optuna"]
                                    if state["status"] == "interrupted" and previous
                                    else trial["optuna"]
                                    if not previous
                                    else session.ask(
                                        trial["index"], parameters=trial["parameters"]
                                    )[1]
                                )
                            write_json(attempt / "attempt.json", attempt_record)
                        process = context.Process(
                            target=_worker,
                            args=(
                                str(directory),
                                str(attempt),
                                trial,
                                record["execution_identity"],
                                primitive(options) if options else None,
                                attempt_record.get("optuna"),
                                messages,
                            ),
                        )
                        process.start()
                        active[trial["id"]] = (process, attempt, attempt_record)
                        emit(trial["id"], {"stage": "starting worker"})
                    while not messages.empty():
                        name, payload = messages.get()
                        emit(
                            name,
                            payload,
                            total=payload.get("display_total"),
                            unit=payload.get("display_unit"),
                        )
                        if on_progress:
                            on_progress(
                                {
                                    "trial_id": name,
                                    **{
                                        k: v
                                        for k, v in payload.items()
                                        if not k.startswith("display_")
                                    },
                                }
                            )
                    for name, (process, attempt, attempt_record) in list(
                        active.items()
                    ):
                        if process.is_alive():
                            continue
                        process.join()
                        if not (attempt / "result.json").exists():
                            _atomic(
                                attempt / "result.json",
                                {
                                    "status": "interrupted",
                                    "runs": list(_saved_runs(attempt).values()),
                                    "value": None,
                                    "failure": {
                                        "type": "WorkerLost",
                                        "exitcode": process.exitcode,
                                    },
                                },
                            )
                        result = read_json(attempt / "result.json")
                        emit(
                            name,
                            {
                                "stage": result["status"],
                                "status": result["status"],
                                "reason": (
                                    "Validation incomplete; candidate ineligible"
                                    if result["status"] == "ineligible"
                                    else (result.get("failure") or {}).get(
                                        "message",
                                        (result.get("failure") or {}).get("type", ""),
                                    )
                                ),
                            },
                            final=True,
                        )
                        if session:
                            session.finish(
                                attempt_record["optuna"],
                                result["status"],
                                result.get("value"),
                            )
                        del active[name]
                        _summary(root, record, interrupted=interrupted)
                    if interrupted and not active:
                        break
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    if interrupted:
                        for process, _, _ in active.values():
                            if process.is_alive():
                                os.killpg(process.pid, signal.SIGINT)
                    interrupted = True
        finally:
            for process, _, _ in active.values():
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGINT)
                process.join()
            messages.close()
        return _summary(root, record, interrupted=interrupted)


@operation("batch-train")
def run_learning_batch(
    configs=None,
    *,
    output_root=None,
    max_concurrent=1,
    resume=None,
    retry_failed=False,
    on_progress=None,
):
    if resume is not None:
        if configs is not None or output_root is not None:
            raise ConfigurationError(
                "resume uses the saved batch plan; do not provide new configs/output_root"
            )
        return _execute(resume, retry_failed=retry_failed, on_progress=on_progress)
    if not configs or type(max_concurrent) is not int or max_concurrent < 1:
        raise ConfigurationError("batch requires configs and a positive max_concurrent")
    trials = [_trial((config,), index) for index, config in enumerate(configs)]
    plan = {
        "schema": PLAN_SCHEMA,
        "kind": "learning_batch",
        "max_concurrent": max_concurrent,
        "trials": trials,
    }
    root, record = _allocate(
        plan, output_root or configs[0].output.root, "learning-batch"
    )
    record["trial_budget"] = len(trials)
    _atomic(root / "run.json", record)
    return _execute(root, on_progress=on_progress)


@operation("search")
def search(config=None, *, resume=None, retry_failed=False, on_progress=None):
    if resume is not None:
        if config is not None:
            raise ConfigurationError("resume uses the saved search configuration")
        return _execute(resume, retry_failed=retry_failed, on_progress=on_progress)
    if config is None:
        raise ConfigurationError("search requires a configuration")
    validate_search(config)
    if config.search.method == "optuna":
        require_optuna()
        trials = []
        templates = [_freeze(item) for item in trial_configs(config, {})]
        budget = config.search.trials
    else:
        templates = []
        trials = [
            _trial(trial_configs(config, values), index, values)
            for index, values in enumerate(candidates(config))
        ]
        budget = len(trials)
    plan = {
        "schema": PLAN_SCHEMA,
        "kind": "search",
        "base": primitive(config),
        "max_concurrent": config.runtime.max_concurrent,
        "trials": trials,
        "templates": templates,
    }
    root, record = _allocate(plan, config.output.root, f"{config.output.name}-search")
    record.update(trial_budget=budget, direction=config.search.direction)
    _atomic(root / "run.json", record)
    return _execute(root, on_progress=on_progress)
