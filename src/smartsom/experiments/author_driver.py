"""Task-aware native plans, stage ledgers and existing measured Tune dispatch."""

import json
import multiprocessing
import os
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
from smartsom.experiments.control import (
    PauseRequested,
    StopRequested,
    boundary,
    requested,
)
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.telemetry.runtime import CURRENT, bind, operation, worker_output

SCHEMA = "smartsom.author-plan/v1"
DONE = {"completed", "early_stopped"}


def _entry_workflow(prepared, task):
    from smartsom.telemetry.workflow import describe_prepared

    metadata = describe_prepared(
        prepared,
        {
            "train": "training",
            "evaluate": "evaluation",
            "train-evaluate": "train-evaluate",
        }[task],
    )
    metadata["control_names"] = (
        ["initial"]
        if task == "train-evaluate" and prepared.config.evaluation.initial_control
        else []
    )
    return metadata


def allocate(plan, *, root=None):
    """Only execution allocates; check never reaches this boundary."""
    root = (
        Path(root)
        if root is not None
        else Path(plan.experiment.output.root)
        / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + plan.experiment.output.name
            + "-"
            + uuid4().hex[:10]
        )
    )
    root.mkdir(parents=True, exist_ok=False)
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
            if (
                entry["task"] == "train-evaluate"
                and prepared.config.evaluation.initial_control
            ):
                required.add("initial_control")
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
                        or record.get("kind")
                        != ("evaluation" if phase == "initial_control" else phase)
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
    display = CURRENT.get()
    if display:
        display.kind = {
            "train": "training",
            "evaluate": "evaluation",
            "train-evaluate": "train-evaluate",
        }[entry["task"]]
        display.configure(_entry_workflow(prepared, entry["task"]))
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
        if display and ledger["status"] in DONE:
            display.update(
                "evaluation" if entry["task"] != "train" else "training",
                {"stage": "finalizing"},
            )
        write_json(path, ledger)
        if display:
            stages = ledger["stages"]
            completed = [
                phase
                for stage, phases in {
                    "training": ["training", "validation"],
                    "evaluation": ["evaluation"],
                    "initial_control": ["initial"],
                }.items()
                if stages.get(stage, {}).get("status") in DONE
                for phase in phases
            ]
            ticks = None
            training = stages.get("training", {})
            if training.get("status") in DONE and training.get("run_dir"):
                ticks = json.loads(
                    (Path(training["run_dir"]) / "run.json").read_text()
                ).get("physical_ticks")
            display.seed_execution(physical_ticks=ticks, completed_phases=completed)
            if ledger["status"] in DONE:
                display.update(
                    "evaluation" if entry["task"] != "train" else "training",
                    {"stage": ledger["status"], "status": ledger["status"]},
                    final=True,
                )

    def source_for(stage):
        source = stage.get("run_dir")
        if source and not Path(source).resolve().is_relative_to(worker_dir.resolve()):
            raise ValueError("stage directory escapes its entry")
        return Path(source) if source else None

    try:
        save()  # seed completed stages before a resumed final evaluation begins
        boundary(worker_dir)
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
                        "paused"
                        if isinstance(exc, PauseRequested)
                        else "stopped"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    )
                    save()
                    raise
        boundary(worker_dir)
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
                        result = evaluate(
                            prepared,
                            purpose=purpose,
                            resume_dir=output / "final-evaluation",
                        )
                    else:
                        source = Path(ledger["stages"]["training"]["run_dir"])
                        from smartsom.experiments.composable import evaluate

                        result = evaluate(
                            source=source,
                            selection=prepared.config.evaluation.checkpoint,
                            output_root=output,
                            resume_dir=output / "final-evaluation",
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
                        "paused"
                        if isinstance(exc, PauseRequested)
                        else "stopped"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    )
                    save()
                    raise
        if (
            entry["task"] == "train-evaluate"
            and prepared.config.evaluation.initial_control
        ):
            boundary(worker_dir)
            stage = ledger["stages"].setdefault(
                "initial_control", {"status": "pending", "attempts": []}
            )
            if stage["status"] not in DONE:
                stage["status"] = "running"
                save()
                try:
                    from smartsom.experiments.composable import evaluate

                    source = Path(ledger["stages"]["training"]["run_dir"])
                    result = evaluate(
                        source=source,
                        selection="initial"
                        if prepared.config.checkpointing.retention is not None
                        else "update-000000",
                        output_root=output,
                        purpose="initial_control",
                        resume_dir=output / "initial-control",
                    )
                    stage["attempts"].append(str(result.run_dir))
                    stage.update(status=result.status, run_dir=str(result.run_dir))
                    if result.status in DONE:
                        stage["manifest_sha256"] = digest(
                            json.loads((result.run_dir / "run.json").read_text())
                        )
                    save()
                    if result.status not in DONE:
                        raise RuntimeError("initial control has engineering failures")
                except BaseException as exc:
                    if getattr(exc, "run_dir", None):
                        stage["attempts"].append(str(exc.run_dir))
                    stage["status"] = (
                        "paused"
                        if isinstance(exc, PauseRequested)
                        else "stopped"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    )
                    save()
                    raise
        ledger["status"] = "completed"
    except PauseRequested:
        ledger["status"] = "paused"
    except KeyboardInterrupt:
        ledger["status"] = "stopped"
    except BaseException as exc:
        ledger.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        save()
    return ledger


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
    algorithm_name = Path(entry["sources"]["algorithm"]).stem
    event = {
        "status": status,
        "stage": status,
        "display_name": f"{algorithm_name} · seed {prepared.config.seed} ({entry['id']})",
    }
    workflow = _entry_workflow(prepared, task)
    snapshot = root / "entries" / entry["id"] / "logs/progress.json"
    if snapshot.is_file():
        try:
            payload = json.loads(snapshot.read_text())
            rows = payload["tasks"]
            from smartsom.telemetry.execution_metrics import source_execution

            event["execution"] = source_execution(payload)
            latest = max(rows, key=lambda r: r.get("updated_at", 0), default={})
            event.update(latest.get("values", {}))
            if (
                event["execution"]
                and event["execution"].get("physical_ticks") is not None
            ):
                event["physical_ticks"] = event["execution"]["physical_ticks"]
            event["source_updated_at"] = latest.get(
                "updated_at", payload.get("updated_at")
            )
            event["stage"] = latest.get("stage", status) if not final else status
            event["status"] = status
        except (OSError, ValueError, KeyError):
            pass
    from smartsom.telemetry.execution_metrics import merged_plan

    # Frozen obligations exist before worker startup. Child phase overlays must
    # not turn a train-evaluate plan into its current training/evaluation phase.
    event["workflow"] = merged_plan(event.get("workflow"), workflow)
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


def _native_source_admission(root):
    """Native directory entries share the source gate, just like Tune actors."""
    from smartsom.experiments import control, queue_admission

    scope = control.CURRENT.get()
    source = Path(scope.driver_root) if scope and scope.driver_root else None
    if (
        source is None
        or source == root
        or not (source / "control/admission.json").is_file()
    ):
        return None
    source_plan = json.loads((source / "plan.json").read_text())
    if source_plan.get("schema") != "smartsom.author-batch-plan/v1":
        return None  # an appended job already owns its queue's native slot
    journal = queue_admission.snapshot(source)
    if journal.get("schema") != queue_admission.SCHEMA:
        return None
    file = next(
        (
            file
            for file in source_plan["files"]
            if (source / file["child"]).resolve() == root.resolve()
        ),
        None,
    )
    if file is not None and "native:" + file["id"] in journal.get("slots", {}):
        return None
    from smartsom.experiments.runtime_control import snapshot

    runtime = snapshot(source)
    native_jobs = {**runtime.get("tasks", {}), **runtime.get("additions", {})}
    for key, job in native_jobs.items():
        if (
            job.get("root")
            and Path(job["root"]).resolve() == root.resolve()
            and "native:" + key in journal.get("slots", {})
        ):
            return None  # outer scheduler already owns this one-experiment slot
    if file is None:
        raise ValueError("native child is absent from its frozen directory plan")
    return source, journal["owner_id"], file["id"]


def _native_source_reserve(
    admission,
    entry_id,
    *,
    cpus=1,
    memory=320 * 1024**2,
    observed=None,
    resource_model=None,
):
    if admission is None:
        return True
    from smartsom.experiments.queue_admission import note_wait, reserve, snapshot

    source, owner, file_id = admission
    if resource_model is not None:
        from smartsom.experiments.runtime_resources import estimate

        prediction = estimate(
            source,
            resource_model["descriptor"],
            cpus=resource_model["cpus"],
            memory=resource_model["memory"],
        )
        cpus = max(cpus, prediction["cpus"])
        memory = max(memory, int(prediction["memory"] * 1.25))
    reserve_ram = (
        max(2 * 1024**3, int(observed.memory_total * 0.2))
        if observed is not None
        else 0
    )
    note_wait(
        source,
        "source:" + file_id + "__" + entry_id,
        owner_id=owner,
        cpus=cpus,
        memory=memory,
        host=observed,
    )
    if observed is not None and (
        observed.cpus * (1 - observed.cpu_load / 100) < cpus + 0.25
        or observed.memory_available < reserve_ram + memory
    ):
        return False
    from smartsom.experiments.runtime_control import snapshot as runtime_snapshot

    runtime = runtime_snapshot(source)
    if runtime.get("mode") == "adaptive" and len(
        snapshot(source)["slots"]
    ) >= runtime.get("native_effective_limit", runtime["limit"]):
        return False
    return reserve(
        source,
        "source:" + file_id + "__" + entry_id,
        owner_id=owner,
        status="pending",
        cpus=cpus,
        memory=memory,
        resource_model=resource_model,
        host=observed,
    )


def _native_source_started(admission, entry_id):
    if admission is not None:
        from types import SimpleNamespace

        from smartsom.experiments.queue_admission import synchronize

        source, owner, file_id = admission
        synchronize(
            source,
            {file_id + "__" + entry_id: SimpleNamespace(state="running")},
            owner_id=owner,
        )


def _native_source_retire(admission, entry_id):
    if admission is not None:
        from smartsom.experiments.queue_admission import retire

        source, owner, file_id = admission
        retire(source, "source:" + file_id + "__" + entry_id, owner_id=owner)


def _entry_resource_observation(row, *, memory=None, cpu=None, inactive=False):
    """Project existing samples, independently of admission resource estimates."""
    previous = row.get("resources", {})
    row["resources"] = dict(
        cpu_cores=cpu if memory is not None else None,
        memory=memory,
        memory_peak=max(
            previous.get("memory_peak") or 0, row.get("peak_memory", 0), memory or 0
        )
        or None,
        observed_at=time.time() if memory is not None else previous.get("observed_at"),
        status="inactive"
        if inactive
        else "current"
        if memory is not None
        else "unavailable",
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

        execution = plan["experiment"]["execution"]
        session = CURRENT.get()
        if session and (
            execution.get("tuning", execution.get("performance", "off")) != "off"
            or execution["executor"] == "tune"
        ):
            session.kind = "tune"
            session.configure_tuning(
                {
                    **(previous_tuning or {}),
                    "calibration": {
                        **((previous_tuning or {}).get("calibration") or {}),
                        "level": execution.get("calibration_level", "quick"),
                    },
                }
            )
        try:
            run_preflight(root, plan)
        except PauseRequested:
            state["status"] = "paused"
            _save(root, state)
            return {
                "run_directory": str(root),
                "status": "paused",
                "entries": state["entries"],
            }
        if (
            execution.get("tuning", execution.get("performance", "off")) != "off"
            or execution["executor"] == "tune"
        ):
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
        admission = _native_source_admission(root)
        native_observer = None
        native_resources, native_peaks = {}, {}
        native_descriptors = {}
        if admission is not None:
            from smartsom.experiments import runtime_resources
            from smartsom.experiments.tuning_resources import ResourceMonitor

            native_observer = ResourceMonitor()
            resource_sampler = runtime_resources.TreeSampler()
            for entry in plan["entries"]:
                frozen = prepared_from_run(root / entry["snapshot"])
                native_descriptors[entry["id"]] = runtime_resources.describe(
                    asdict(frozen), task=entry["task"]
                )
                settings = frozen.config.runtime
                native_resources[entry["id"]] = settings.numerical_threads * (
                    1 + settings.sampling_processes
                )
                native_peaks[entry["id"]] = state["entries"][entry["id"]].get(
                    "peak_memory", 0
                )
        pending, active = [], {}
        for entry in plan["entries"]:
            path = _ledger_path(root, entry)
            if path.exists():
                ledger = json.loads(path.read_text())
                previous = state["entries"][entry["id"]]
                if previous.get("resources"):
                    ledger["resources"] = previous["resources"]
                state["entries"][entry["id"]] = ledger
            row = state["entries"][entry["id"]]
            _entry_resource_observation(row, inactive=True)
            if row["status"] == "completed":
                _progress(session, root, entry, "completed", final=True)
            else:
                pending.append(entry)
                _progress(session, root, entry, "queued")
        state["status"] = "running"
        _save(root, state)
        context = multiprocessing.get_context("spawn")
        stopping = False
        pausing = False
        inner_pause_sent, inner_stop_sent = set(), set()
        try:
            while pending or active:
                from smartsom.experiments.control import pause_requested

                pausing = pause_requested(root)
                native_observation = (
                    native_observer.snapshot(exclude_pids=(os.getpid(),))
                    if native_observer
                    else None
                )
                inner = None
                if admission is not None:
                    from smartsom.experiments import runtime_control as runtime
                    from smartsom.experiments.control import read as read_owner
                    from smartsom.experiments.control import request_pause
                    from smartsom.experiments.control import write_json as control_write

                    source, _, file_id = admission
                    if runtime.snapshot(source):
                        with runtime.transaction(source) as saved:
                            for entry in plan["entries"]:
                                saved["tasks"][file_id + "__" + entry["id"]] = dict(
                                    state["entries"][entry["id"]],
                                    file_id=file_id,
                                    root=str(root / "entries" / entry["id"]),
                                )
                        inner = runtime.apply(source)
                        for entry in plan["entries"]:
                            key = entry["id"]
                            identity = file_id + "__" + key
                            if (
                                state["entries"][key]["status"] == "stopped"
                                and key not in active
                                and entry not in pending
                                and not runtime.matches(
                                    identity, inner["stopped_tasks"]
                                )
                                and not runtime.matches(
                                    identity, inner.get("removed_tasks", [])
                                )
                            ):
                                state["entries"][key]["status"] = "queued"
                                pending.append(entry)
                        for entry in list(pending):
                            key = entry["id"]
                            identity = file_id + "__" + key
                            if runtime.matches(
                                identity, inner.get("removed_tasks", [])
                            ):
                                state["entries"][key]["status"] = "removed"
                                pending.remove(entry)
                            elif runtime.matches(identity, inner["stopped_tasks"]):
                                state["entries"][key]["status"] = "stopped"
                                pending.remove(entry)
                            elif runtime.matches(identity, inner["paused_tasks"]):
                                state["entries"][key]["status"] = "paused"
                            elif state["entries"][key]["status"] == "paused":
                                state["entries"][key]["status"] = "queued"
                        for key, (process, _) in active.items():
                            identity = file_id + "__" + key
                            folder = root / "entries" / key
                            if runtime.matches(
                                identity,
                                inner["stopped_tasks"] + inner.get("removed_tasks", []),
                            ):
                                owner = read_owner(folder)
                                if owner and key not in inner_stop_sent:
                                    control_write(
                                        folder / "control/stop.json",
                                        {"id": owner["id"]},
                                    )
                                    inner_stop_sent.add(key)
                            elif (
                                runtime.matches(identity, inner["paused_tasks"])
                                and key not in inner_pause_sent
                            ):
                                try:
                                    request_pause(folder)
                                except ValueError:
                                    pass
                                else:
                                    inner_pause_sent.add(key)
                if requested(root):
                    stopping = True
                    state["status"] = "stopping"
                    _save(root, state)
                # Shared admission is authoritative for directory matrices. The
                # frozen standalone cap must not prevent a live limit increase.
                admission_limit = inner["limit"] if inner is not None else concurrency
                while (
                    pending
                    and len(active) < admission_limit
                    and not stopping
                    and not pausing
                ):
                    entry = None
                    candidates = (
                        sorted(
                            pending,
                            key=lambda entry: runtime.priority_rank(
                                inner, file_id + "__" + entry["id"]
                            ),
                        )
                        if inner is not None
                        else pending
                    )
                    for candidate in candidates:
                        if inner is not None and runtime.matches(
                            file_id + "__" + candidate["id"], inner["paused_tasks"]
                        ):
                            continue
                        if _native_source_reserve(
                            admission,
                            candidate["id"],
                            cpus=native_resources.get(candidate["id"], 1),
                            memory=int(
                                max(256 * 1024**2, native_peaks.get(candidate["id"], 0))
                                * 1.25
                            ),
                            observed=native_observation,
                            resource_model=dict(
                                descriptor=native_descriptors[candidate["id"]],
                                cpus=native_resources[candidate["id"]],
                                memory=max(
                                    256 * 1024**2, native_peaks.get(candidate["id"], 0)
                                ),
                            )
                            if admission is not None
                            else None,
                        ):
                            entry = candidate
                            state["entries"][entry["id"]].pop("blocking_reason", None)
                            break
                        state["entries"][candidate["id"]]["blocking_reason"] = (
                            "waiting for shared CPU/memory budget, resource headroom, or older feasible task"
                        )
                    if entry is None:
                        break
                    pending.remove(entry)
                    folder = root / "entries" / entry["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    process = context.Process(
                        target=_worker, args=(str(root), entry, str(folder))
                    )
                    process.start()
                    active[entry["id"]] = (process, entry)
                    _native_source_started(admission, entry["id"])
                    state["entries"][entry["id"]]["status"] = "running"
                    if native_observation is not None:
                        # A launch consumes headroom from this observation. Sample
                        # again before considering another shared native worker.
                        break
                for key, (process, entry) in list(active.items()):
                    if process.is_alive():
                        _entry_resource_observation(state["entries"][key])
                        if admission is not None:
                            import psutil

                            from smartsom.experiments.queue_admission import (
                                update_resources,
                            )

                            try:
                                child = psutil.Process(process.pid)
                                peak = sum(
                                    p.memory_info().rss
                                    for p in [child, *child.children(recursive=True)]
                                )
                                native_peaks[key] = max(peak, native_peaks.get(key, 0))
                                state["entries"][key]["peak_memory"] = native_peaks[key]
                                folder = root / "entries" / key
                                descriptor = native_descriptors[key]
                                phase = runtime_resources.stage_at(folder)
                                sampled_memory, sampled_cpu = resource_sampler.sample(
                                    process.pid, phase
                                )
                                _entry_resource_observation(
                                    state["entries"][key],
                                    memory=sampled_memory
                                    if sampled_memory is not None
                                    else peak,
                                    cpu=sampled_cpu,
                                )
                                runtime_resources.observe(
                                    source,
                                    descriptor,
                                    phase,
                                    memory=sampled_memory,
                                    cpus=sampled_cpu,
                                )
                                runtime_resources.ingest(source, descriptor, folder)
                                prediction = runtime_resources.estimate(
                                    source,
                                    descriptor,
                                    cpus=native_resources[key],
                                    memory=max(256 * 1024**2, native_peaks[key]),
                                )
                                state["entries"][key]["resource_estimate"] = prediction
                                update_resources(
                                    source,
                                    "source:" + file_id + "__" + key,
                                    owner_id=admission[1],
                                    cpus=prediction["cpus"],
                                    resource_estimate=prediction,
                                    observed_memory=sampled_memory,
                                    observed_cpus=sampled_cpu,
                                    observed_processes=resource_sampler.identities.get(
                                        process.pid, ()
                                    ),
                                    memory=int(prediction["memory"] * 1.25),
                                )
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        _progress(
                            session, root, entry, "stopping" if stopping else "running"
                        )
                        continue
                    process.join()
                    if admission is not None:
                        runtime_resources.ingest(
                            source, native_descriptors[key], root / "entries" / key
                        )
                    _native_source_retire(admission, entry["id"])
                    path = _ledger_path(root, entry)
                    observations = state["entries"][key].get("resources", {})
                    state["entries"][key] = (
                        json.loads(path.read_text())
                        if path.exists()
                        else {
                            "status": "failed",
                            "error": f"worker exited {process.exitcode} before stage commit",
                        }
                    )
                    state["entries"][key]["resources"] = observations
                    _entry_resource_observation(state["entries"][key], inactive=True)
                    if process.exitcode:
                        state["entries"][key].update(
                            status="failed", error=f"worker exited {process.exitcode}"
                        )
                    if key in native_peaks:
                        state["entries"][key]["peak_memory"] = native_peaks[key]
                    if inner is not None and not process.exitcode:
                        identity = file_id + "__" + key
                        if state["entries"][key][
                            "status"
                        ] != "failed" and runtime.matches(
                            identity, inner.get("removed_tasks", [])
                        ):
                            state["entries"][key]["status"] = "removed"
                        elif (
                            state["entries"][key]["status"] == "paused" and not pausing
                        ):
                            pending.append(entry)
                    inner_pause_sent.discard(key)
                    inner_stop_sent.discard(key)
                    _progress(
                        session,
                        root,
                        entry,
                        state["entries"][key]["status"],
                        final=True,
                    )
                    del active[key]
                _save(root, state)
                if (stopping or pausing) and not active:
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
                _native_source_retire(admission, entry["id"])
                path = _ledger_path(root, entry)
                observations = state["entries"][key].get("resources", {})
                state["entries"][key] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {
                        "status": "failed",
                        "error": "worker exited without a stage ledger",
                    }
                )
                state["entries"][key]["resources"] = observations
                _entry_resource_observation(state["entries"][key], inactive=True)
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
            "failed"
            if any(s == "failed" for s in statuses)
            else "stopped"
            if stopping
            else "paused"
            if pausing or any(s == "paused" for s in statuses)
            else "partial_removed"
            if any(s == "removed" for s in statuses)
            else "stopped"
            if any(s not in {"completed", "failed"} for s in statuses)
            else "completed"
        )
        _save(root, state)
        if admission is not None:
            from smartsom.experiments import runtime_control as runtime
            from smartsom.experiments.queue_admission import clear_wait

            source, _, file_id = admission
            for entry in plan["entries"]:
                clear_wait(
                    source,
                    "source:" + file_id + "__" + entry["id"],
                    owner_id=admission[1],
                )
            if runtime.snapshot(source):
                with runtime.transaction(source) as saved:
                    for key, row in state["entries"].items():
                        saved["tasks"][file_id + "__" + key] = dict(
                            row, file_id=file_id
                        )
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

                prepared = PreparedComposition(**saved["selected_prepared"])
                event["workflow"] = _entry_workflow(prepared, "train-evaluate")
                authoring = json.loads(prepared.training_inputs_json).get(
                    "authoring", {}
                )
                algorithm_path = authoring.get("sources", {}).get("algorithm")
                if algorithm_path:
                    event["display_name"] = (
                        f"{Path(algorithm_path).stem} · seed {prepared.config.seed} ({key})"
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
            active_limit=(
                settings["calibration_seconds"]
                if settings.get("calibration_seconds") is not None
                else 0.0
                if settings.get("calibration_level") in {"off", "online"}
                else 1800.0
                if settings.get("calibration_level") == "full"
                else 300.0
            ),
            calibration_level=settings.get("calibration_level", "quick"),
            calibration_candidate=settings.get("calibration_candidate", "latest"),
            output_root=str(root / "performance"),
            provenance={
                "kind": "author-plan",
                "plan_sha256": digest(plan),
                "performance_root": plan["experiment"]["output"]["root"],
            },
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
