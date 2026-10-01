"""Task-aware native plans, stage ledgers and existing measured Tune dispatch."""

import json
import multiprocessing
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.experiments.composable import (
    archive_inputs,
    implementation_identity,
    prepared_from_run,
    verify_prepared_rules,
)
from smartsom.experiments.control import StopRequested, boundary, requested
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.telemetry.runtime import CURRENT, bind, operation, worker_output

SCHEMA = "smartsom.author-plan/v1"
DONE = {"completed", "early_stopped"}


def allocate(plan):
    """Only execution allocates; check never reaches this boundary."""
    root = Path(plan.experiment.output.root) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + plan.experiment.output.name
        + "-"
        + uuid4().hex[:10]
    )
    root.mkdir(parents=True)
    entries = []
    for e in plan.entries:
        target = root / "inputs" / e.id
        (target / "config").mkdir(parents=True)
        frozen = archive_inputs(target, e.prepared)
        entries.append(
            {
                "id": e.id,
                "task": e.task,
                "snapshot": str(target.relative_to(root)),
                "prepared_sha256": digest(asdict(frozen)),
                "scientific_sha256": frozen.scientific_sha256,
                "sources": e.sources,
                "H": e.heterogeneity,
                "workload": e.workload,
            }
        )
    saved = {
        "schema": SCHEMA,
        "experiment": primitive(plan.experiment),
        "entries": entries,
        "authoring": plan.authoring,
        "source": source_identity(),
        "implementation_sha256": implementation_identity(),
    }
    write_json(root / "plan.json", saved)
    state = {
        "schema": "smartsom.author-state/v1",
        "plan_sha256": digest(saved),
        "status": "prepared",
        "entries": {e["id"]: {"status": "queued"} for e in entries},
    }
    write_json(root / "batch.json", state)
    write_json(
        root / "run.json",
        {
            "schema": "smartsom.author-run/v1",
            "kind": "plan",
            "status": "prepared",
            "plan_sha256": digest(saved),
            "source": saved["source"],
            "id": root.name,
        },
    )
    return root


def load(root):
    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if plan.get("schema") != SCHEMA or digest(plan) != state["plan_sha256"]:
        raise ValueError("frozen author plan changed")
    if implementation_identity() != plan["implementation_sha256"]:
        raise ValueError("implementation identity changed; create a new run")
    live = source_identity()
    if (
        live["python"] != plan["source"]["python"]
        or live["packages"] != plan["source"]["packages"]
        or live["git"]["commit"] != plan["source"]["git"]["commit"]
    ):
        raise ValueError("source/dependency identity changed; create a new run")
    delegated = state.get("tune_directory")
    if delegated:
        from smartsom.experiments.tuning_batch import load_run

        directory = Path(delegated).resolve()
        if not directory.is_relative_to(root / "performance"):
            raise ValueError("Tune directory escapes author run")
        _, tune_plan, tune_state = load_run(directory)
        if tune_plan.get("provenance", {}).get("plan_sha256") != digest(plan):
            raise ValueError("Tune plan does not belong to this author run")
        if {e["experiment_id"] for e in tune_plan["entries"]} != {
            e["id"] for e in plan["entries"]
        }:
            raise ValueError("Tune entry identities changed")
        for key, row in state["entries"].items():
            if row["status"] == "completed":
                committed = tune_state["entries"][key]
                if committed["status"] != "completed" or not committed.get(
                    "checkpoint"
                ):
                    raise ValueError(
                        "completed Tune entry lacks committed stage evidence"
                    )
                checkpoint = Path(committed["checkpoint"]).resolve()
                if not checkpoint.is_relative_to(directory):
                    raise ValueError("Tune checkpoint escapes its run")
                from smartsom.config.experiment_v3 import PreparedComposition
                from smartsom.experiments.tuning_session import verify_identity

                frozen = next(
                    e for e in tune_plan["entries"] if e["experiment_id"] == key
                )
                marker = verify_identity(
                    PreparedComposition(**frozen["prepared"]),
                    json.loads((checkpoint / "record.json").read_text()),
                    checkpoint,
                )
                if marker["phase"] != "experiment_complete":
                    raise ValueError(
                        "completed Tune entry lacks final evaluation commit"
                    )
    for entry in plan["entries"]:
        target = (root / entry["snapshot"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("input snapshot escapes run directory")
        prepared = prepared_from_run(target)
        if digest(asdict(prepared)) != entry["prepared_sha256"]:
            raise ValueError("frozen author input changed")
        if delegated:
            continue  # Tune validates its own committed training/evaluation stages.
        ledger_file = _ledger_path(root, entry)
        if (
            state["entries"].get(entry["id"], {}).get("status") == "completed"
            and not ledger_file.exists()
        ):
            raise ValueError("completed entry stage evidence is missing")
        if ledger_file.exists():
            ledger = json.loads(ledger_file.read_text())
            if ledger.get("scientific_sha256") != prepared.scientific_sha256:
                raise ValueError("stage ledger scientific identity changed")
            required = (
                {"training"}
                if entry["task"] == "train"
                else {"evaluation"}
                if entry["task"] == "evaluate"
                else {"training", "evaluation"}
            )
            if ledger.get("status") == "completed" and (
                not required <= ledger.get("stages", {}).keys()
                or any(
                    ledger["stages"][phase].get("status") not in DONE
                    for phase in required
                )
            ):
                raise ValueError("completed entry lacks required stages")
            for phase, stage in ledger.get("stages", {}).items():
                if phase == "training" and stage.get("run_dir"):
                    directory = Path(stage["run_dir"]).resolve()
                    if not directory.is_relative_to(root / "entries" / entry["id"]):
                        raise ValueError("training stage directory escapes entry")
                    record = json.loads((directory / "run.json").read_text())
                    if (
                        record.get("kind") != "training"
                        or record.get("scientific_sha256") != prepared.scientific_sha256
                    ):
                        raise ValueError("training stage scientific identity changed")
                if stage.get("status") in DONE:
                    directory = Path(stage.get("run_dir", "")).resolve()
                    if not directory.is_relative_to(root / "entries" / entry["id"]):
                        raise ValueError("completed stage directory escapes entry")
                    record_file = directory / "run.json"
                    if not record_file.is_file():
                        raise ValueError("completed stage artifacts are missing")
                    record = json.loads(record_file.read_text())
                    if (
                        record.get("status") not in DONE
                        or record.get("kind") != phase
                        or stage.get("manifest_sha256") != digest(record)
                    ):
                        raise ValueError("completed stage evidence changed")
                    if (
                        phase == "training"
                        and record.get("scientific_sha256")
                        != prepared.scientific_sha256
                    ):
                        raise ValueError(
                            "completed training scientific identity changed"
                        )
    return root, plan, state


def _save(root, state):
    write_json(root / "batch.json", state)
    record = json.loads((root / "run.json").read_text())
    record["status"] = state["status"]
    record["entries"] = state["entries"]
    write_json(root / "run.json", record)


def _ledger_path(root, entry):
    return root / "entries" / entry["id"] / "stages.json"


@worker_output("worker_dir")
@operation("author-entry")
def _worker(root, entry, worker_dir):
    from smartsom import api

    root, worker_dir = Path(root), Path(worker_dir)
    from smartsom.experiments.control import CURRENT as CONTROL

    CONTROL.get().inherit_driver(root)
    bind(worker_dir, entry["id"])
    prepared = prepared_from_run(root / entry["snapshot"])
    verify_prepared_rules(prepared)
    path = _ledger_path(root, entry)
    ledger = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "status": "running",
            "stages": {},
            "scientific_sha256": prepared.scientific_sha256,
        }
    )
    if ledger["scientific_sha256"] != prepared.scientific_sha256:
        raise ValueError("stage ledger scientific identity changed")
    data = json.loads(prepared.config_json)
    output = worker_dir / "runs"
    data["output"]["root"] = str(output)
    prepared = replace(prepared, config_json=canonical_json(data))

    def save():
        write_json(path, ledger)

    def source_for(stage):
        source = stage.get("run_dir")
        if source and not Path(source).resolve().is_relative_to(worker_dir.resolve()):
            raise ValueError("stage directory escapes its entry")
        return Path(source) if source else None

    try:
        boundary(root)
        if entry["task"] != "evaluate":
            stage = ledger["stages"].setdefault("training", {"status": "pending"})
            source = source_for(stage)
            if stage["status"] not in DONE:
                # Reconcile a committed child whose parent died before ledger commit.
                if source is None:
                    candidates = [
                        p.parent
                        for p in output.glob("*/run.json")
                        if json.loads(p.read_text()).get("kind") == "training"
                        and json.loads(p.read_text()).get("scientific_sha256")
                        == prepared.scientific_sha256
                    ]
                    source = sorted(candidates)[-1] if candidates else None
                stage["status"] = "running"
                save()
                try:
                    if (
                        source
                        and json.loads((source / "run.json").read_text())["status"]
                        in DONE
                    ):
                        result = json.loads((source / "run.json").read_text())
                        status = result["status"]
                    else:
                        result = (
                            api.resume(source)
                            if source
                            and (source / "checkpoints/recovery.json").is_file()
                            else api.train_prepared(prepared)
                        )
                        source, status = result.run_dir, result.status
                    stage.update(status=status, run_dir=str(source))
                    if status in DONE:
                        stage["manifest_sha256"] = digest(
                            json.loads((source / "run.json").read_text())
                        )
                    save()
                    if status not in DONE:
                        raise StopRequested("training stopped at its saved update")
                except BaseException as exc:
                    if getattr(exc, "run_dir", None):
                        stage["run_dir"] = str(exc.run_dir)
                    stage["status"] = (
                        "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
                    )
                    save()
                    raise
        boundary(root)
        if entry["task"] != "train":
            stage = ledger["stages"].setdefault(
                "evaluation", {"status": "pending", "attempts": []}
            )
            if stage["status"] not in DONE:
                stage["status"] = "running"
                save()
                try:
                    if entry["task"] == "evaluate":
                        from smartsom.experiments.composable import evaluate

                        purpose = (
                            json.loads(prepared.training_inputs_json)
                            .get("authoring", {})
                            .get("purpose")
                        )
                        result = evaluate(prepared, purpose=purpose)
                    else:
                        source = Path(ledger["stages"]["training"]["run_dir"])
                        from smartsom.experiments.composable import evaluate

                        result = evaluate(
                            source=source,
                            selection=prepared.config.evaluation.checkpoint,
                            output_root=output,
                        )
                    stage["attempts"].append(str(result.run_dir))
                    stage.update(status=result.status, run_dir=str(result.run_dir))
                    if result.status in DONE:
                        stage["manifest_sha256"] = digest(
                            json.loads((result.run_dir / "run.json").read_text())
                        )
                    save()
                    if result.status not in DONE:
                        raise RuntimeError("evaluation has engineering failures")
                except BaseException as exc:
                    if getattr(exc, "run_dir", None):
                        stage["attempts"].append(str(exc.run_dir))
                    stage["status"] = (
                        "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
                    )
                    save()
                    raise
        ledger["status"] = "completed"
    except KeyboardInterrupt:
        ledger["status"] = "stopped"
    except BaseException as exc:
        ledger.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        save()


def _progress(session, root, entry, status, *, final=False):
    if session is None:
        return
    prepared = prepared_from_run(root / entry["snapshot"])
    task = entry["task"]
    total = (
        prepared.config.training.total_ticks
        if task != "evaluate"
        else len(json.loads(prepared.evaluation_json))
    )
    event = {"status": status, "stage": status, "display_name": entry["id"]}
    snapshot = root / "entries" / entry["id"] / "logs/progress.json"
    if snapshot.is_file():
        try:
            rows = json.loads(snapshot.read_text())["tasks"]
            latest = max(rows, key=lambda r: r.get("updated_at", 0), default={})
            event.update(latest.get("values", {}))
            event["stage"] = latest.get("stage", status) if not final else status
            event["status"] = status
        except (OSError, ValueError, KeyError):
            pass
    event["context"] = (
        f"{task} | envs={prepared.config.runtime.num_envs} sampling={prepared.config.runtime.sampling_processes} threads={prepared.config.runtime.numerical_threads} device={prepared.config.runtime.device}"
    )
    session.update(
        entry["id"],
        event,
        total=total,
        unit="physical ticks" if task != "evaluate" else "evaluation episodes",
        final=final,
    )


@operation("author-plan")
def execute_saved(root, *, max_concurrent=None):
    """Resume frozen stages; scientific input changes always require new allocation."""
    from smartsom.experiments.batch import exclusive_lock

    root, plan, state = load(root)
    previous_tuning = None
    progress = root / "logs/progress.json"
    if progress.is_file():
        previous_tuning = json.loads(progress.read_text()).get("tuning")
    bind(root)
    with exclusive_lock(root / "driver.lock"):
        from smartsom.experiments.preflight import run as run_preflight

        run_preflight(root, plan)
        execution = plan["experiment"]["execution"]
        if (
            execution.get("tuning", execution.get("performance", "off")) != "off"
            or execution["executor"] == "tune"
        ):
            if CURRENT.get() and previous_tuning:
                CURRENT.get().kind = "tune"
                CURRENT.get().configure_tuning(previous_tuning)
            if max_concurrent is not None:
                raise ValueError(
                    "Tune concurrency is selected by calibration; native concurrency overrides do not apply"
                )
            state["status"] = "running"
            _save(root, state)
            try:
                return _tune(root, plan, state)
            except BaseException as exc:
                state.update(
                    status="stopped"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if state.get("tune_directory"):
                    state["entries"] = json.loads(
                        (Path(state["tune_directory"]) / "batch.json").read_text()
                    )["entries"]
                _save(root, state)
                _publish_tune_state(CURRENT.get(), state)
                raise
        concurrency = (
            execution["max_concurrent"] if max_concurrent is None else max_concurrent
        )
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError("max_concurrent must be a positive integer")
        session = CURRENT.get()
        if session:
            session.total_tasks = len(plan["entries"])
        pending, active = [], {}
        for entry in plan["entries"]:
            path = _ledger_path(root, entry)
            if path.exists():
                ledger = json.loads(path.read_text())
                state["entries"][entry["id"]] = ledger
            row = state["entries"][entry["id"]]
            if row["status"] == "completed":
                _progress(session, root, entry, "completed", final=True)
            else:
                pending.append(entry)
                _progress(session, root, entry, "queued")
        state["status"] = "running"
        _save(root, state)
        context = multiprocessing.get_context("spawn")
        stopping = False
        try:
            while pending or active:
                if requested(root):
                    stopping = True
                    state["status"] = "stopping"
                    _save(root, state)
                while pending and len(active) < concurrency and not stopping:
                    entry = pending.pop(0)
                    folder = root / "entries" / entry["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    process = context.Process(
                        target=_worker, args=(str(root), entry, str(folder))
                    )
                    process.start()
                    active[entry["id"]] = (process, entry)
                    state["entries"][entry["id"]]["status"] = "running"
                for key, (process, entry) in list(active.items()):
                    if process.is_alive():
                        _progress(
                            session, root, entry, "stopping" if stopping else "running"
                        )
                        continue
                    process.join()
                    path = _ledger_path(root, entry)
                    state["entries"][key] = (
                        json.loads(path.read_text())
                        if path.exists()
                        else {
                            "status": "failed",
                            "error": f"worker exited {process.exitcode} before stage commit",
                        }
                    )
                    _progress(
                        session,
                        root,
                        entry,
                        state["entries"][key]["status"],
                        final=True,
                    )
                    del active[key]
                _save(root, state)
                if stopping and not active:
                    break
                time.sleep(0.15)
        except BaseException as exc:
            from smartsom.experiments.control import read
            from smartsom.experiments.control import write_json as control_write

            owner = read(root)
            control_write(root / "control/stop.json", {"id": owner["id"]})
            for process, _ in active.values():
                process.join()  # cooperative safe boundary; explicit force is separate
            for key, (_, entry) in active.items():
                path = _ledger_path(root, entry)
                state["entries"][key] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {
                        "status": "failed",
                        "error": "worker exited without a stage ledger",
                    }
                )
                _progress(
                    session, root, entry, state["entries"][key]["status"], final=True
                )
            stopping = True
            if not isinstance(exc, KeyboardInterrupt):
                state["status"] = "failed"
                state["error"] = f"{type(exc).__name__}: {exc}"
                _save(root, state)
                raise
        statuses = [row["status"] for row in state["entries"].values()]
        state["status"] = (
            "stopped"
            if stopping or any(s not in {"completed", "failed"} for s in statuses)
            else "failed"
            if any(s == "failed" for s in statuses)
            else "completed"
        )
        _save(root, state)
        return {
            "run_directory": str(root),
            "status": state["status"],
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "pending": sum(s not in {"completed", "failed"} for s in statuses),
            "entries": state["entries"],
        }


def _publish_tune_state(display, state):
    """Final saved phase state supersedes asynchronous broker presentation."""
    if display is None:
        return
    from smartsom.telemetry.runtime import FINAL

    summary = display.tuning or {}
    previous = {row["experiment_id"]: row for row in summary.get("entries", ())}
    rows = []
    with display.batch_updates():
        for key, saved in state["entries"].items():
            status = (
                "recommended" if state["status"] == "recommended" else saved["status"]
            )
            rows.append(
                {**previous.get(key, {}), "experiment_id": key, "status": status}
            )
            event = {
                "status": status,
                "stage": status,
                "physical_ticks": saved.get("physical_ticks", 0),
            }
            if saved.get("selected_prepared"):
                from smartsom.config.experiment_v3 import PreparedComposition
                from smartsom.telemetry.workflow import describe_prepared

                event["workflow"] = describe_prepared(
                    PreparedComposition(**saved["selected_prepared"]),
                    "train-evaluate",
                )
            if status == "completed" and saved.get("attempts"):
                attempt = Path(saved["attempts"][-1]["run_dir"])
                progress_path = attempt / "logs/progress.json"
                if progress_path.is_file():
                    training = next(
                        (
                            item
                            for item in json.loads(progress_path.read_text())["tasks"]
                            if item["id"] == "training"
                        ),
                        None,
                    )
                    if training:
                        for name in (
                            "validation_finished",
                            "validation_requested",
                            "validation_batches_finished",
                        ):
                            if name in training.get("values", {}):
                                event[name] = training["values"][name]
                evaluation_path = attempt / "evaluation/summary.json"
                if evaluation_path.is_file():
                    evaluation = json.loads(evaluation_path.read_text())
                    event["evaluation_requested"] = evaluation["requested"]
                    event["evaluation_finished"] = sum(
                        evaluation[name]
                        for name in ("completed", "truncated", "exceptions")
                    )
            display.update(
                key,
                event,
                final=status in FINAL,
            )
        display.configure_tuning({**summary, "stage": state["status"], "entries": rows})


def _tune(root, plan, state):
    from smartsom.experiments.tuning_batch import (
        BatchInputs,
        allocate_batch,
        execute_batch,
        load_run,
        preflight,
    )

    display = CURRENT.get()
    if display:
        display.kind = "tune"

    if state.get("tune_directory"):
        directory, tune_plan, tune_state = load_run(state["tune_directory"])
        result = execute_batch(
            directory,
            tune_plan,
            tune_state,
            recommend_only=plan["experiment"]["execution"].get(
                "tuning", plan["experiment"]["execution"].get("performance")
            )
            == "recommend",
            display=display,
        )
    else:
        entries = tuple(
            {
                "experiment_id": e["id"],
                "prepared": asdict(prepared_from_run(root / e["snapshot"])),
                "control_spec": {},
                "baseline_concurrency": plan["experiment"]["execution"][
                    "max_concurrent"
                ],
            }
            for e in plan["entries"]
        )
        settings = plan["experiment"]["execution"]
        inputs = BatchInputs(
            entries,
            mode=settings["mode"],
            execution=settings["scheduling"],
            active_limit=settings["calibration_seconds"],
            output_root=str(root / "performance"),
            provenance={"kind": "author-plan", "plan_sha256": digest(plan)},
        )
        preflight(inputs)
        directory, tune_plan, tune_state = allocate_batch(inputs)
        state["tune_directory"] = str(directory)
        _save(root, state)
        result = execute_batch(
            directory,
            tune_plan,
            tune_state,
            recommend_only=settings.get("tuning", settings.get("performance"))
            == "recommend",
            display=display,
        )
    state["status"] = result.get("status", "completed")
    state["entries"] = json.loads(
        (Path(state["tune_directory"]) / "batch.json").read_text()
    )["entries"]
    _save(root, state)
    _publish_tune_state(display, state)
    return {"run_directory": str(root), **result}


def run(plan, *, display_options=None):
    root = allocate(plan)
    if plan.experiment.execution.background:
        from smartsom.experiments.background import launch

        return launch(root, display_options=display_options)
    return execute_saved(root, display_options=display_options)
