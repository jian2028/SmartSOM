"""A frozen directory of V4 Experiments with one parent lifecycle.

Directory scheduling is execution metadata. Each child is still compiled by
the ordinary V4 compiler and retains its own scientific identity and inputs.
"""

import json
import multiprocessing
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.experiments.composable import implementation_identity
from smartsom.experiments.control import requested
from smartsom.experiments.evidence import (
    runtime_source_matches,
    source_identity,
    write_json,
)
from smartsom.telemetry.runtime import CURRENT, bind, operation
from smartsom.telemetry.runtime import FINAL as DISPLAY_FINAL

SCHEMA = "smartsom.author-batch-plan/v1"
STATE_SCHEMA = "smartsom.author-batch-state/v1"
RUN_SCHEMA = "smartsom.author-batch-run/v1"
CALIBRATION_SECONDS = {"online": 0, "off": 0, "quick": 5 * 60, "full": 30 * 60}
FINAL = {"completed", "failed", "stopped"}


@dataclass(frozen=True)
class DirectoryPlan:
    directory: Path
    files: tuple
    output_root: Path
    calibration_seconds: float
    calibration_level: str = "quick"
    calibration_candidate: str = "latest"
    selection: dict | None = None
    contract: str = "v4"

    def summary(self):
        if self.contract == "reservation":
            return {
                "status": "checked",
                "input_type": "reservation-directory",
                "directory": str(self.directory),
                "output_root": str(self.output_root),
                "count_files": len(self.files),
                "files": [
                    {
                        "id": row["id"],
                        "stage": row["stage"],
                        **row["compiled"].summary(),
                    }
                    for row in self.files
                ],
            }
        return {
            "status": "checked",
            "input_type": "experiment-v4-directory",
            "directory": str(self.directory),
            "output_root": str(self.output_root),
            "calibration_seconds": self.calibration_seconds,
            "calibration_level": self.calibration_level,
            "calibration_candidate": self.calibration_candidate,
            "count_files": len(self.files),
            "count_entries": sum(len(row["compiled"].entries) for row in self.files),
            **({"selection": self.selection} if self.selection is not None else {}),
            "files": [
                {
                    "id": row["id"],
                    "path": str(row["path"]),
                    "stage": row["stage"],
                    "parallel_files": row["parallel_files"],
                    "gate": row["gate"],
                    **({"requires": row["requires"]} if row.get("requires") else {}),
                    "task": row["compiled"].experiment.task,
                    "count": len(row["compiled"].entries),
                    "entries": [
                        {"id": e.id, "scientific_sha256": e.prepared.scientific_sha256}
                        for e in row["compiled"].entries
                    ],
                }
                for row in self.files
            ],
            "scope": "input checks only; no run directory, learner, Ray or calibration",
        }


def compile_directory(
    directory,
    *,
    require_dependencies=False,
    calibration_seconds=None,
    calibration_level=None,
    calibration_candidate=None,
    ids=None,
    tags=None,
):
    """Check every direct YAML member, failing on an unsupported or duplicate input."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigurationError(
            f"batch-run requires an Experiment directory: {directory}"
        )
    if calibration_level is not None and calibration_level not in CALIBRATION_SECONDS:
        raise ConfigurationError("calibration level must be online, off, quick or full")
    if calibration_seconds is not None and (
        not isinstance(calibration_seconds, (int, float))
        or isinstance(calibration_seconds, bool)
        or not 0 <= calibration_seconds < float("inf")
    ):
        raise ConfigurationError(
            "batch calibration budget must be finite and nonnegative"
        )
    calibration_overrides = {
        key: value
        for key, value in (
            ("calibration_level", calibration_level),
            ("calibration_candidate", calibration_candidate),
            ("calibration_seconds", calibration_seconds),
        )
        if value is not None
    }
    if calibration_seconds == 0:
        # V4 represents no isolated timeout as None; the shared budget below
        # still validates that an explicit zero is only used with online/off.
        calibration_overrides["calibration_seconds"] = None
    paths = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        ),
        key=lambda p: p.name,
    )
    if not paths:
        raise ConfigurationError(
            "batch-run directory contains no Experiment YAML files"
        )
    inventory = {}
    for path in paths:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", path.stem):
            raise ConfigurationError(f"unsafe batch Experiment name: {path.name}")
        if path.stem in inventory:
            raise ConfigurationError(f"duplicate batch Experiment name: {path.stem}")
        inventory[path.stem] = path
    if ids is not None:
        if not ids or len(set(ids)) != len(ids):
            raise ConfigurationError("batch IDs must be nonempty and unique")
        unknown = set(ids) - set(inventory)
        if unknown:
            raise ConfigurationError(
                "unknown batch Experiment IDs: " + ", ".join(sorted(unknown))
            )
        paths = [inventory[identity] for identity in ids]
    if tags is not None:
        if not tags or any(not tag for tag in tags) or len(set(tags)) != len(tags):
            raise ConfigurationError("batch tags must be nonempty and unique")
        from smartsom.config.experiment_v4 import _read_source

        matched = []
        for path in paths:
            document, _ = _read_source(path)
            output = document.get("output", {})
            if not isinstance(output, dict):
                raise ConfigurationError(f"{path.name}: output must be a mapping")
            authored = output.get("tags", [])
            if not isinstance(authored, list) or any(
                type(tag) is not str for tag in authored
            ):
                raise ConfigurationError(
                    f"{path.name}: output.tags must be a list of strings"
                )
            if set(tags) <= set(authored):
                matched.append(path)
            elif ids is not None:
                raise ConfigurationError(
                    f"{path.stem}: requested ID does not match all tags"
                )
        paths = matched
        if not paths:
            raise ConfigurationError("batch selection matched no Experiments")
    from smartsom.config.reservation import is_reservation

    reservation_inputs = [is_reservation(path) for path in paths]
    if any(reservation_inputs):
        if not all(reservation_inputs):
            raise ConfigurationError(
                "mixing Base and V4 contracts in one batch is unsupported"
            )
        from smartsom.experiments.reservation_batch import compile_files

        return compile_files(directory, paths, calibration_overrides)
    files, scientific = [], set()
    output_root = None
    for index, path in enumerate(paths):
        try:
            compiled = compile_experiment(
                path,
                require_dependencies=require_dependencies,
                batch_calibration=calibration_overrides,
            )
        except Exception as exc:
            raise ConfigurationError(f"{path.name}: {exc}") from exc
        if output_root is None:
            output_root = Path(compiled.experiment.output.root)
        elif output_root != Path(compiled.experiment.output.root):
            raise ConfigurationError("batch Experiments must share one output.root")
        for entry in compiled.entries:
            if entry.prepared.scientific_sha256 in scientific:
                raise ConfigurationError(
                    f"duplicate scientific entry in batch directory: {path.name}/{entry.id}"
                )
            scientific.add(entry.prepared.scientific_sha256)
        metadata = compiled.experiment.batch
        if metadata and metadata.gate and compiled.experiment.task != "evaluate":
            raise ConfigurationError(f"{path.name}: batch gate requires evaluate")
        files.append(
            {
                "id": path.stem,
                "path": path,
                "compiled": compiled,
                "stage": metadata.stage if metadata else index,
                "parallel_files": metadata.parallel_files if metadata else 1,
                "gate": primitive(metadata.gate)
                if metadata and metadata.gate
                else None,
                "annotated": metadata is not None,
                **(
                    {"requires": list(metadata.requires)}
                    if metadata and metadata.requires
                    else {}
                ),
            }
        )
    authored_ids = {row["id"] for row in files}
    generated_ids = [
        row["id"] + "__" + entry.id
        for row in files
        for entry in row["compiled"].entries
    ]
    if authored_ids.intersection(generated_ids) or len(generated_ids) != len(
        set(generated_ids)
    ):
        raise ConfigurationError("batch task ID collides with a generated entry ID")
    if any(row["annotated"] for row in files) and not all(
        row["annotated"] for row in files
    ):
        raise ConfigurationError(
            "batch.stage must be present in every Experiment or none"
        )
    for stage in {row["stage"] for row in files}:
        limits = {row["parallel_files"] for row in files if row["stage"] == stage}
        if len(limits) != 1:
            raise ConfigurationError(
                f"batch stage {stage} has conflicting parallel_files"
            )
    learning = [
        row for row in files if row["compiled"].experiment.task == "train-evaluate"
    ]
    if learning:
        stages = {row["stage"] for row in learning}
        settings = {
            (
                row["compiled"].experiment.execution.mode,
                row["compiled"].experiment.execution.scheduling,
                row["compiled"].experiment.execution.preflight,
                row["compiled"].experiment.execution.preflight_coverage,
            )
            for row in learning
        }
        if len(settings) != 1:
            raise ConfigurationError(
                "learning files must agree on shared tuning settings"
            )
        if any(
            row["compiled"].experiment.task != "train-evaluate"
            for row in files
            if row["stage"] in stages
        ):
            raise ConfigurationError(
                "shared learning stage cannot contain rule/native files"
            )
    settings = learning[0]["compiled"].experiment.execution if learning else None
    level = calibration_level or (settings.calibration_level if settings else "quick")
    if learning and level == "online" and settings.scheduling != "adaptive":
        raise ConfigurationError("online feedback requires adaptive scheduling")
    candidate = calibration_candidate or (
        settings.calibration_candidate if settings else "latest"
    )
    if (
        learning
        and calibration_level is None
        and any(
            row["compiled"].experiment.execution.calibration_level != level
            for row in learning
        )
    ):
        raise ConfigurationError("learning files must agree on calibration level")
    if (
        learning
        and calibration_candidate is None
        and any(
            row["compiled"].experiment.execution.calibration_candidate != candidate
            for row in learning
        )
    ):
        raise ConfigurationError("learning files must agree on calibration candidate")
    if calibration_seconds is None and learning:
        values = {
            row["compiled"].experiment.execution.calibration_seconds for row in learning
        }
        if len(values) != 1:
            raise ConfigurationError("learning files must agree on calibration timeout")
        calibration_seconds = values.pop()
    if calibration_seconds is None:
        calibration_seconds = CALIBRATION_SECONDS[level]
    if (
        not isinstance(calibration_seconds, (int, float))
        or isinstance(calibration_seconds, bool)
        or not 0 <= calibration_seconds < float("inf")
        or (level in {"off", "online"} and calibration_seconds != 0)
        or (level not in {"off", "online"} and calibration_seconds == 0)
    ):
        raise ConfigurationError(
            "batch calibration budget must be zero only when online or off"
        )
    rank = {path.stem: index for index, path in enumerate(paths)}
    files.sort(
        key=lambda row: (
            row["stage"],
            rank[row["id"]] if ids is not None else row["id"],
        )
    )
    by_id = {row["id"]: row for row in files}
    for row in files:
        dependencies = row.get("requires", [])
        if len(set(dependencies)) != len(dependencies):
            raise ConfigurationError(f"{row['id']}: duplicate batch.requires")
        for required in dependencies:
            if required not in by_id:
                raise ConfigurationError(
                    f"{row['id']}: missing selected dependency {required}"
                )
            if by_id[required]["stage"] >= row["stage"]:
                raise ConfigurationError(
                    f"{row['id']}: dependency {required} must be in an earlier stage"
                )
    selection = None
    if ids is not None or tags is not None:
        selection = {
            "ids": list(ids) if ids is not None else None,
            "tags": list(tags) if tags is not None else None,
            "resolved_ids": [row["id"] for row in files],
        }
    return DirectoryPlan(
        directory,
        tuple(files),
        output_root,
        float(calibration_seconds),
        level,
        candidate,
        selection,
    )


def _tune_inputs(directory_plan, *, output_root, parent_digest, stage=None):
    from smartsom.experiments.tuning_batch import BatchInputs

    learning = [
        r
        for r in directory_plan.files
        if r["compiled"].experiment.task == "train-evaluate"
        and (stage is None or r["stage"] == stage)
    ]
    if not learning:
        return None
    setting = learning[0]["compiled"].experiment.execution
    entries = tuple(
        {
            "experiment_id": f"{row['id']}__{entry.id}",
            "file_id": row["id"],
            "prepared": asdict(entry.prepared),
            "control_spec": {},
            "baseline_concurrency": row["compiled"].experiment.execution.max_concurrent,
        }
        for row in learning
        for entry in row["compiled"].entries
    )
    return BatchInputs(
        entries,
        mode=setting.mode,
        execution=setting.scheduling,
        active_limit=directory_plan.calibration_seconds,
        calibration_level=directory_plan.calibration_level,
        calibration_candidate=directory_plan.calibration_candidate,
        output_root=str(output_root),
        provenance={
            "kind": "author-batch",
            "parent_plan_sha256": parent_digest,
            "performance_root": str(directory_plan.output_root),
            **({"stage": stage} if stage is not None else {}),
        },
        preflight=setting.preflight,
        preflight_coverage=setting.preflight_coverage,
    )


def allocate(directory_plan):
    """Freeze the whole batch and children before starting any control or learner."""
    if directory_plan.contract == "reservation":
        from smartsom.experiments.reservation_batch import allocate_directory

        return allocate_directory(directory_plan)
    from smartsom.experiments.author_driver import allocate as allocate_author
    from smartsom.experiments.tuning_batch import allocate_batch, preflight

    root = directory_plan.output_root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-batch-"
        + directory_plan.directory.name
        + "-"
        + uuid4().hex[:10]
    )
    # Check framework dependencies and all learning entries before allocating.
    tentative = _tune_inputs(
        directory_plan, output_root=root / "performance", parent_digest="preflight"
    )
    if tentative:
        preflight(tentative)
    root.mkdir(parents=True, exist_ok=False)
    saved_files = []
    for row in directory_plan.files:
        child = root / "experiments" / row["id"]
        allocate_author(row["compiled"], root=child)
        child_plan = json.loads((child / "plan.json").read_text())
        saved_files.append(
            {
                "id": row["id"],
                "path": str(row["path"]),
                "source_sha256": json.loads(
                    row["compiled"].entries[0].prepared.training_inputs_json
                )["authoring"]["input_sha256"][str(row["path"])],
                "child": str(child.relative_to(root)),
                "child_plan_sha256": digest(child_plan),
                "entry_ids": [e.id for e in row["compiled"].entries],
                "scientific_sha256": [
                    e.prepared.scientific_sha256 for e in row["compiled"].entries
                ],
                "task": row["compiled"].experiment.task,
                "stage": row["stage"],
                "parallel_files": row["parallel_files"],
                "gate": row["gate"],
                **({"requires": row["requires"]} if row.get("requires") else {}),
            }
        )
    plan = {
        "schema": SCHEMA,
        "directory": str(directory_plan.directory),
        "files": saved_files,
        "calibration_seconds": directory_plan.calibration_seconds,
        "calibration_level": directory_plan.calibration_level,
        "calibration_candidate": directory_plan.calibration_candidate,
        "source": source_identity(),
        "implementation_sha256": implementation_identity(),
        **(
            {"selection": directory_plan.selection}
            if directory_plan.selection is not None
            else {}
        ),
    }
    write_json(root / "plan.json", plan)
    tune_directory = None
    learning_stages = sorted(
        {
            row["stage"]
            for row in directory_plan.files
            if row["compiled"].experiment.task == "train-evaluate"
        }
    )
    stage_scoped = len({row["stage"] for row in directory_plan.files}) > 1
    tune_directories = {}
    if tentative:
        for stage in learning_stages:
            inputs = _tune_inputs(
                directory_plan,
                output_root=root / "performance" / f"stage-{stage}"
                if stage_scoped
                else root / "performance",
                parent_digest=digest(plan),
                stage=stage if stage_scoped else None,
            )
            created, _, _ = allocate_batch(inputs)
            tune_directories[str(stage)] = str(created.relative_to(root))
        if len(learning_stages) == 1:
            tune_directory = next(iter(tune_directories.values()))
    state = {
        "schema": STATE_SCHEMA,
        "plan_sha256": digest(plan),
        "status": "prepared",
        "stage": "prepared",
        "files": {row["id"]: {"status": "queued"} for row in saved_files},
        "tune_directory": tune_directory,
        "calibration_status": (
            "online"
            if directory_plan.calibration_level == "online" and tentative
            else "skipped"
            if directory_plan.calibration_level == "off"
            else "queued"
            if tentative
            else "not_applicable"
        ),
        "smoke_status": "queued",
        **(
            {
                "tune_directories": tune_directories,
                "calibration_by_stage": {
                    str(stage): "online"
                    if directory_plan.calibration_level == "online"
                    else "skipped"
                    if directory_plan.calibration_level == "off"
                    else "queued"
                    for stage in learning_stages
                },
            }
            if stage_scoped and tentative
            else {}
        ),
    }
    write_json(root / "batch.json", state)
    write_json(
        root / "run.json",
        {
            "schema": RUN_SCHEMA,
            "kind": "batch-directory",
            "name": root.name,
            "status": "prepared",
            "stage": "prepared",
            "plan_sha256": digest(plan),
            "source": plan["source"],
        },
    )
    return root


def load(root):
    from smartsom.experiments.author_driver import load as load_author
    from smartsom.experiments.tuning_batch import load_run

    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if (
        plan.get("schema") != SCHEMA
        or state.get("schema") != STATE_SCHEMA
        or digest(plan) != state.get("plan_sha256")
    ):
        raise ValueError("frozen V4 directory batch changed")
    if (
        not runtime_source_matches(plan["source"], source_identity())
        or implementation_identity() != plan["implementation_sha256"]
    ):
        raise ValueError("batch source/dependency identity changed; create a new run")
    for row in plan["files"]:
        child = (root / row["child"]).resolve()
        if not child.is_relative_to(root / "experiments"):
            raise ValueError("batch child escapes its run")
        if plan.get("contract") == "reservation":
            from smartsom.experiments.reservation import load as load_reservation

            child_plan = load_reservation(child)
        else:
            _, child_plan, _ = load_author(child)
        if digest(child_plan) != row["child_plan_sha256"]:
            raise ValueError("frozen child plan changed")
        if (
            plan.get("contract") != "reservation"
            and [e["scientific_sha256"] for e in child_plan["entries"]]
            != row["scientific_sha256"]
        ):
            raise ValueError("batch scientific entry identity changed")
    tune_paths = state.get("tune_directories")
    if tune_paths is not None:
        expected_stages = {
            str(row["stage"])
            for row in plan["files"]
            if row["task"] == "train-evaluate"
        }
        if set(tune_paths) != expected_stages or len(set(tune_paths.values())) != len(
            tune_paths
        ):
            raise ValueError("learning stage Tune identities changed")
        if (
            state.get("tune_directory") is not None
            and state["tune_directory"] not in tune_paths.values()
        ):
            raise ValueError("active Tune directory is not a frozen learning stage")
    else:
        tune_paths = (
            {None: state["tune_directory"]} if state.get("tune_directory") else {}
        )
    for stage, location in tune_paths.items():
        child = (root / location).resolve()
        if not child.is_relative_to(root / "performance"):
            raise ValueError("batch Tune directory escapes its run")
        _, tune_plan, _ = load_run(child)
        if tune_plan.get("provenance", {}).get("parent_plan_sha256") != digest(plan):
            raise ValueError("Tune plan belongs to a different directory batch")
        if stage is not None and tune_plan.get("provenance", {}).get("stage") != int(
            stage
        ):
            raise ValueError("Tune plan belongs to a different learning stage")
        expected = {
            f"{row['id']}__{entry}"
            for row in plan["files"]
            if row["task"] == "train-evaluate"
            and (stage is None or row["stage"] == int(stage))
            for entry in row["entry_ids"]
        }
        if {e["experiment_id"] for e in tune_plan["entries"]} != expected:
            raise ValueError("shared Tune entry identities changed")
    return root, plan, state


def _save(root, state):
    write_json(root / "batch.json", state)
    record = json.loads((root / "run.json").read_text())
    record.update(status=state["status"], stage=state["stage"], files=state["files"])
    write_json(root / "run.json", record)


def _smoke_all(root, plan, state, *, stage=None):
    """Exercise each frozen entry before the shared performance clock starts."""
    if plan.get("contract") == "reservation":
        return  # Base inputs are compiled before allocation; no V4 smoke is applicable.
    from smartsom.experiments.composable import prepared_from_run
    from smartsom.experiments.preflight import smoke

    if state.get("smoke_status") == "completed":
        return
    saved = root / "smoke.json"
    evidence = (
        json.loads(saved.read_text())
        if saved.exists()
        else {
            "schema": "smartsom.author-batch-smoke/v1",
            "entries": {},
            "status": "running",
            "started_at": time.time(),
            "tick_limit": 128,
        }
    )
    state.update(stage="smoke", smoke_status="running")
    _save(root, state)
    files = [row for row in plan["files"] if stage is None or row["stage"] == stage]
    selected_ids = {
        f"{row['id']}__{entry}" for row in files for entry in row["entry_ids"]
    }
    total = len(selected_ids)

    def show():
        if CURRENT.get():
            CURRENT.get().configure_preflight(
                {
                    "status": "passed"
                    if evidence["status"] == "completed"
                    else evidence["status"],
                    "level": "full",
                    "coverage": "each",
                    "checks_done": total,
                    "checks_total": total,
                    "smoke_done": len(selected_ids & evidence["entries"].keys()),
                    "smoke_total": total,
                    "smoke_skipped": 0,
                    "current": evidence.get("current"),
                    "entries": {
                        key: value
                        for key, value in evidence["entries"].items()
                        if key in selected_ids
                    },
                }
            )

    show()
    for row in files:
        child = root / row["child"]
        child_plan = json.loads((child / "plan.json").read_text())
        for entry in child_plan["entries"]:
            identity = f"{row['id']}__{entry['id']}"
            if evidence["entries"].get(identity, {}).get("status") == "passed":
                continue
            if requested(root):
                raise KeyboardInterrupt("batch stopped during smoke")
            prepared = prepared_from_run(child / entry["snapshot"])
            from smartsom.experiments.queue_admission import smoke_slot

            with smoke_slot(root, identity):
                outcome = smoke(prepared, cancelled=lambda: requested(root))
            evidence["entries"][identity] = outcome
            evidence["current"] = identity
            write_json(saved, evidence)
            show()
            if outcome["status"] != "passed":
                evidence["status"] = "failed"
                write_json(saved, evidence)
                show()
                raise RuntimeError(f"{identity}: smoke {outcome}")
    all_ids = {
        f"{row['id']}__{entry}" for row in plan["files"] for entry in row["entry_ids"]
    }
    complete = all_ids <= evidence["entries"].keys()
    evidence.update(
        status="completed" if complete else "running", finished_at=time.time()
    )
    write_json(saved, evidence)
    show()
    state["smoke_status"] = "completed" if complete else "running"
    _save(root, state)


@operation("tune")
def _tune_worker(root, *, recommend_only, retry_failed=False):
    from smartsom.experiments.tuning_batch import execute_batch, load_run

    root, plan, state = load_run(root)
    bind(root)
    if CURRENT.get():
        CURRENT.get().kind = "tune"
    return execute_batch(
        root,
        plan,
        state,
        recommend_only=recommend_only,
        retry_failed=retry_failed,
        display=CURRENT.get(),
    )


def _quiet_worker(
    root, parent_root, *, tuning=False, recommend_only=False, retry_failed=False
):
    """Keep child diagnostics in child logs while the parent owns the Rich view."""
    from smartsom.experiments.control import CURRENT as CONTROL
    from smartsom.experiments.control import Scope

    root = Path(root)
    scope = Scope()
    scope.inherit_driver(parent_root)
    token = CONTROL.set(scope)
    status = "failed"
    try:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        with (
            (root / "logs/driver-stdout.log").open("a") as stdout,
            (root / "logs/driver-stderr.log").open("a") as stderr,
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                if tuning:
                    result = _tune_worker(
                        root,
                        recommend_only=recommend_only,
                        retry_failed=retry_failed,
                        display_options={"progress": "off", "verbose": False},
                    )
                else:
                    schema = json.loads((root / "plan.json").read_text()).get("schema")
                    if schema == "smartsom.reservation-plan/v1":
                        from smartsom.experiments.reservation import execute_saved

                        result = execute_saved(root)
                        if result["status"] in {"frozen", "wall_budget_reached"}:
                            # Scheduler completion is not scientific completion.
                            # Keep the native outcome/counters and original deadline.
                            write_json(
                                root / "batch.json",
                                {
                                    **result,
                                    "status": "completed",
                                    "native_status": result["status"],
                                },
                            )
                    else:
                        from smartsom.experiments.author_driver import execute_saved

                        result = execute_saved(
                            root, display_options={"progress": "off", "verbose": False}
                        )
        status = result.get("status", "completed")
        return result
    except KeyboardInterrupt:
        status = "stopped"
        raise
    finally:
        scope.finish(status)
        CONTROL.reset(token)


def _child_snapshot(root):
    path = root / "logs/progress.json"
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def _last_error_line(message):
    lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
    return lines[-1][:240] if lines else ""


def _tune_failure(state):
    for identity, row in state.get("entries", {}).items():
        if row.get("status") == "failed":
            failure = row.get("failure") or {}
            message = _last_error_line(failure.get("message"))
            if message:
                return f"{identity}: {message}"
    return _last_error_line((state.get("failure") or {}).get("message"))


def _publish(root, plan, state):
    session = CURRENT.get()
    if session is None:
        return
    session.configure_batch(
        [
            dict(
                id=row["id"],
                stage=row["stage"],
                barrier_group=str(root),
                requires=row.get("requires", []),
            )
            for row in plan["files"]
        ]
    )
    session.phase(state["stage"])
    tune = root / state["tune_directory"] if state.get("tune_directory") else None
    tune_snapshot = _child_snapshot(tune) if tune else {}
    paths = list((state.get("tune_directories") or {}).values()) or (
        [state["tune_directory"]] if state.get("tune_directory") else []
    )
    by_id, tune_entries = {}, {}
    for location in paths:
        stage_tune = root / location
        by_id.update(
            {row["id"]: row for row in _child_snapshot(stage_tune).get("tasks", ())}
        )
        try:
            tune_entries.update(
                json.loads((stage_tune / "batch.json").read_text()).get("entries", {})
            )
        except (OSError, ValueError):
            pass
    with session.batch_updates():
        for row in plan["files"]:
            key = row["id"]
            status = state["files"][key]["status"]
            if row["task"] == "train-evaluate":
                completed = sum(
                    tune_entries.get(f"{key}__{entry}", {}).get("status") == "completed"
                    for entry in row["entry_ids"]
                )
            else:
                child = root / row["child"] / "batch.json"
                try:
                    child_entries = json.loads(child.read_text()).get("entries", {})
                except (OSError, ValueError):
                    child_entries = {}
                completed = sum(
                    child_entries.get(entry, {}).get("status") == "completed"
                    for entry in row["entry_ids"]
                )
            if plan.get("contract") == "reservation":
                completed = int(status == "completed")
            values = {}
            if row["task"] == "evaluate":
                finished = requested_cases = 0
                active_case = None
                case_counts = []
                observed = 0
                for entry in row["entry_ids"]:
                    entry_root = root / row["child"] / "entries" / entry
                    snapshot = _child_snapshot(entry_root)
                    for task in snapshot.get("tasks", ()):
                        metrics = task.get("values", {})
                        if "evaluation_requested" not in metrics:
                            continue
                        finished += metrics.get("evaluation_finished") or 0
                        requested_cases += metrics.get("evaluation_requested") or 0
                        observed += 1
                        if "evaluation_target" in metrics:
                            case_counts.append(metrics)
                        if metrics.get("evaluation_case_active"):
                            active_case = metrics
                values = {
                    "workflow": {"mode": "evaluation"},
                    "evaluation_finished": finished,
                    "evaluation_requested": requested_cases,
                    "evaluation_case_active": active_case is not None,
                    "evaluation_tick": (active_case or {}).get("evaluation_tick", 0),
                    "evaluation_tick_limit": (active_case or {}).get(
                        "evaluation_tick_limit", 0
                    ),
                }
                if case_counts and len(case_counts) == observed:
                    values.update(
                        evaluation_phase="evaluation",
                        **{
                            name: sum(metrics[name] for metrics in case_counts)
                            for name in (
                                "evaluation_historical_completed",
                                "evaluation_current_completed",
                                "evaluation_remaining",
                                "evaluation_target",
                                "evaluation_remaining_ticks",
                            )
                        },
                    )
            from smartsom.telemetry.execution_metrics import source_execution

            native_execution = (
                source_execution(_child_snapshot(root / row["child"]))
                if row["task"] == "evaluate"
                else None
            )
            session.update(
                key,
                {
                    "execution": native_execution,
                    "display_name": key,
                    "status": status,
                    "stage": status,
                    "context": f"stage {row['stage']} · {row['task']}",
                    "completed": completed,
                    **values,
                },
                total=len(row["entry_ids"]),
                unit="entries",
                final=status in FINAL,
            )
            if row["task"] == "train-evaluate":
                for entry in row["entry_ids"]:
                    identity = f"{key}__{entry}"
                    child = by_id.get(identity, {})
                    tune_entry = tune_entries.get(identity, {})
                    entry_status = tune_entry.get("status", child.get("status", status))
                    failure = tune_entry.get("failure") or {}
                    reason = _last_error_line(failure.get("message"))
                    event = {
                        "execution": child.get("execution"),
                        "source_updated_at": child.get("updated_at"),
                        "status": entry_status,
                        "stage": entry_status
                        if entry_status in DISPLAY_FINAL
                        else child.get("stage", entry_status),
                        "display_name": child.get("name", identity),
                        **child.get("values", {}),
                    }
                    if reason:
                        event["reason"] = reason
                    session.update(
                        identity,
                        event,
                        total=child.get("total"),
                        unit="physical ticks",
                        final=entry_status in DISPLAY_FINAL,
                    )
    if tune:
        tuning = tune_snapshot.get("tuning", {})
        entries = []
        for entry in tuning.get("entries", ()):
            saved_status = tune_entries.get(entry["experiment_id"], {}).get("status")
            # The broker snapshot can predate a final commit. Nonterminal ledger
            # rows may still be queued while their real actor is warming up.
            entries.append(
                {**entry, "status": saved_status}
                if saved_status in DISPLAY_FINAL
                else entry
            )
        learning_active = state["stage"] == "calibration" or any(
            state["stage"] == f"stage_{row['stage']}"
            and row["task"] == "train-evaluate"
            for row in plan["files"]
        )
        session.configure_tuning(
            {
                **tuning,
                "entries": entries,
                "stage": state["stage"],
                "calibration_status": state.get("calibration_status", "queued"),
                "calibration": {
                    **(tuning.get("calibration") or {}),
                    "level": plan.get("calibration_level", "quick"),
                },
                "batch_training_active": learning_active,
            }
        )


def _evaluation_rows(child, entry_ids):
    rows = []
    for entry_id in entry_ids:
        ledger = json.loads((child / "entries" / entry_id / "stages.json").read_text())
        stage = ledger["stages"]["evaluation"]
        if stage["status"] != "completed":
            raise RuntimeError(f"{entry_id}: evaluation was not completed")
        run_dir = Path(stage["run_dir"]).resolve()
        if not run_dir.is_relative_to(child / "entries" / entry_id):
            raise RuntimeError("evaluation artifact escapes its frozen child")
        rows.extend(json.loads((run_dir / "run.json").read_text()).get("results", []))
    return rows


def _gate(root, row):
    child = root / row["child"]
    if row.get("contract") == "reservation":
        path = child / "cases.jsonl"
        cases = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
    else:
        cases = _evaluation_rows(child, row["entry_ids"])
    if any(case.get("engineering_failure") for case in cases):
        raise RuntimeError(f"{row['id']}: engineering failure in rule evaluation")
    gate = row["gate"]
    if gate is None:
        return {"cases": len(cases), "passed": True}
    if len(cases) < gate["min_cases"]:
        raise RuntimeError(
            f"{row['id']}: {len(cases)} cases, expected at least {gate['min_cases']}"
        )
    if any(case.get("delivered", 0) < gate["min_deliveries_each"] for case in cases):
        raise RuntimeError(f"{row['id']}: qualified delivery gate failed")
    if gate["require_first_pickup"] and any(
        case.get("dispatcher_diagnostics", {}).get("first_pickup_tick") is None
        for case in cases
    ):
        raise RuntimeError(f"{row['id']}: first-pickup gate failed")
    return {"cases": len(cases), "passed": True}


def _request_owned_stop(root):
    from smartsom.experiments.control import read
    from smartsom.experiments.control import write_json as control_write

    owner = read(root)
    if owner and owner.get("status") in {"running", "stop_requested", "stopping"}:
        control_write(root / "control/stop.json", {"id": owner["id"]})


def _runtime_initialize(root, plan, state, *, limit=None, keep_open=False):
    from smartsom.experiments import runtime_control as runtime

    default = max((row.get("parallel_files", 1) for row in plan["files"]), default=1)
    # Retain the originally authored execution ceiling until an operator changes it.
    directories = list((state.get("tune_directories") or {}).values())
    if not directories and state.get("tune_directory"):
        directories = [state["tune_directory"]]
    for directory in directories:
        tune_plan = json.loads((root / directory / "plan.json").read_text())
        default = max(
            default,
            max(
                (
                    entry.get("baseline_concurrency", 1)
                    for entry in tune_plan.get("entries", [])
                ),
                default=1,
            ),
        )
    runtime.initialize(
        root,
        limit or default,
        mode="adaptive" if plan.get("calibration_level") == "online" else "manual",
        keep_open=keep_open,
    )
    with runtime.transaction(root) as saved:
        for row in plan["files"]:
            saved["tasks"].setdefault(
                row["id"], {"status": state["files"][row["id"]]["status"]}
            )


def _run_children(root, plan, state, rows, *, parallel_files):
    from smartsom.experiments import runtime_control as runtime
    from smartsom.experiments.runtime_workers import NativeScheduler

    _runtime_initialize(root, plan, state, limit=parallel_files)
    jobs = {
        row["id"]: dict(
            root=str(root / row["child"]),
            status=state["files"][row["id"]]["status"],
            own_slots=len(row.get("entry_ids", [])) > 1,
        )
        for row in rows
    }
    for job in jobs.values():
        if job["status"] not in {"completed", "failed"}:
            job["status"] = "queued"
    scheduler = NativeScheduler(root, jobs)
    try:
        while True:
            if requested(root):
                with runtime.transaction(root) as saved:
                    saved["stopping"] = True
            observed = scheduler.tick()
            for row in rows:
                state["files"][row["id"]].update(jobs[row["id"]])
            state["status"] = observed["status"]
            _save(root, state)
            _publish(root, plan, state)
            if not scheduler.unfinished:
                break
            time.sleep(0.25)
    except BaseException:
        scheduler.close()
        raise
    return not observed["stopping"] and all(
        jobs[row["id"]]["status"]
        in {"completed", "stopped", "removed", "partial_removed", "partial_stopped"}
        for row in rows
    )


def _run_tune(root, plan, state, *, recommend_only, retry_failed=False):
    context = multiprocessing.get_context("spawn")
    tune_root = root / state["tune_directory"]
    process = context.Process(
        target=_quiet_worker,
        args=(str(tune_root), str(root)),
        kwargs={
            "tuning": True,
            "recommend_only": recommend_only,
            "retry_failed": retry_failed,
        },
    )
    process.start()
    from smartsom.experiments.runtime_workers import NativeScheduler

    additions = NativeScheduler(root)
    try:
        while process.is_alive():
            additions.tick()
            _publish(root, plan, state)
            time.sleep(0.5)
        while additions.unfinished:
            additions.tick()
            _publish(root, plan, state)
            time.sleep(0.25)
        process.join()
    except BaseException:
        _request_owned_stop(root)
        process.join()
        additions.close()
        raise
    tune_state = json.loads((tune_root / "batch.json").read_text())
    expected = "recommended" if recommend_only else "completed"
    if requested(root):
        from smartsom.experiments.control import StopRequested

        raise StopRequested("parent batch stopped during shared Tune execution")
    if tune_state["status"] == "stopped":
        from smartsom.experiments.control import StopRequested

        raise StopRequested("learning stage stopped at its saved boundary")
    if process.exitcode or tune_state["status"] not in {
        expected,
        "partial_stopped",
        "partial_removed",
    }:
        failure = _tune_failure(tune_state)
        raise RuntimeError(
            f"shared Tune {'calibration' if recommend_only else 'training'} "
            f"ended {tune_state['status']} (exit {process.exitcode})"
            + (f": {failure}" if failure else "")
        )
    return tune_state


@operation("batch-directory")
def execute_saved(root, *, retry_failed=False):
    from smartsom.experiments.batch import exclusive_lock

    root, plan, state = load(root)
    stage_scoped = len({row["stage"] for row in plan["files"]}) > 1
    bind(root)
    from smartsom.experiments.queue_admission import initialize

    session = CURRENT.get()
    if session:
        session.total_tasks = len(plan["files"]) + sum(
            len(row["entry_ids"])
            for row in plan["files"]
            if row["task"] == "train-evaluate"
        )
        _publish(root, plan, state)
    with exclusive_lock(root / "driver.lock"):
        from smartsom.experiments import runtime_control as runtime

        initialize(root, supported=plan.get("calibration_level") in {"off", "online"})
        _runtime_initialize(root, plan, state)
        if state["status"] in {
            "stopped",
            "failed",
            "partial_stopped",
            "partial_removed",
        }:
            runtime.restart(root)
        try:
            if state["status"] == "completed":
                for row in plan["files"]:
                    if row["task"] == "evaluate":
                        state["files"][row["id"]]["gate"] = _gate(root, row)
                _publish(root, plan, state)
                return {
                    "directory": str(root),
                    "status": "completed",
                    "files": state["files"],
                }
            if state["status"] == "failed" and not retry_failed:
                return {
                    "directory": str(root),
                    "status": "failed",
                    "files": state["files"],
                    "error": state.get("error"),
                    "retry_required": True,
                }
            state.pop("error", None)
            if plan.get("contract") == "reservation" and retry_failed:
                for item in state["files"].values():
                    if item["status"] == "failed":
                        item["status"] = "queued"
            if not stage_scoped:
                _smoke_all(root, plan, state)
            if (
                not stage_scoped
                and state.get("tune_directory")
                and state["calibration_status"]
                not in {
                    "completed",
                    "skipped",
                    "online",
                }
            ):
                state.update(status="running", stage="calibration")
                _save(root, state)
                _publish(root, plan, state)
                _run_tune(root, plan, state, recommend_only=True)
                state["calibration_status"] = "completed"
                _save(root, state)
            stages = sorted({row["stage"] for row in plan["files"]})
            for stage in stages:
                rows = [row for row in plan["files"] if row["stage"] == stage]
                for row in rows:
                    blocked = [
                        identity
                        for identity in row.get("requires", [])
                        if state["files"][identity]["status"] != "completed"
                    ]
                    if blocked:
                        raise RuntimeError(
                            f"{row['id']}: dependencies did not complete: {blocked}"
                        )
                if all(
                    state["files"][row["id"]]["status"] == "completed" for row in rows
                ):
                    for row in rows:
                        if row["task"] == "evaluate":
                            state["files"][row["id"]]["gate"] = _gate(root, row)
                    continue
                if requested(root):
                    state.update(status="stopped", stage="stopped")
                    break
                state.update(status="running", stage=f"stage_{stage}")
                if state.get("tune_directories"):
                    if str(stage) in state["tune_directories"]:
                        state["tune_directory"] = state["tune_directories"][str(stage)]
                        state["calibration_status"] = state["calibration_by_stage"][
                            str(stage)
                        ]
                    else:
                        state["tune_directory"] = None
                _save(root, state)
                _publish(root, plan, state)
                if stage_scoped:
                    _smoke_all(root, plan, state, stage=stage)
                    if rows[0]["task"] == "train-evaluate" and state[
                        "calibration_status"
                    ] not in {"completed", "skipped", "online"}:
                        state.update(stage="calibration")
                        _save(root, state)
                        _run_tune(root, plan, state, recommend_only=True)
                        state["calibration_status"] = "completed"
                        state["calibration_by_stage"][str(stage)] = "completed"
                        _save(root, state)
                    state["stage"] = f"stage_{stage}"
                if rows[0]["task"] == "train-evaluate":
                    for row in rows:
                        state["files"][row["id"]]["status"] = "running"
                    _save(root, state)
                    tune_state = _run_tune(
                        root,
                        plan,
                        state,
                        recommend_only=False,
                        retry_failed=retry_failed,
                    )
                    for row in rows:
                        statuses = [
                            tune_state["entries"][f"{row['id']}__{entry}"]["status"]
                            for entry in row["entry_ids"]
                        ]
                        state["files"][row["id"]]["status"] = (
                            "completed"
                            if all(s == "completed" for s in statuses)
                            else "removed"
                            if all(s == "removed" for s in statuses)
                            else "stopped"
                            if any(s == "stopped" for s in statuses)
                            else "partial_removed"
                            if any(s == "removed" for s in statuses)
                            else "failed"
                        )
                    _save(root, state)
                    if any(
                        state["files"][row["id"]]["status"]
                        not in {"completed", "stopped", "removed", "partial_removed"}
                        for row in rows
                    ):
                        raise RuntimeError(f"learning stage {stage} did not complete")
                else:
                    unfinished = [
                        row
                        for row in rows
                        if state["files"][row["id"]]["status"] != "completed"
                    ]
                    if not _run_children(
                        root,
                        plan,
                        state,
                        unfinished,
                        parallel_files=rows[0]["parallel_files"],
                    ):
                        if requested(root):
                            state.update(status="stopped", stage="stopped")
                            break
                        raise RuntimeError(f"batch stage {stage} did not complete")
                    for row in rows:
                        if (
                            row["task"] == "evaluate"
                            and state["files"][row["id"]]["status"] == "completed"
                        ):
                            state["files"][row["id"]]["gate"] = _gate(root, row)
                            _save(root, state)
            else:
                from smartsom.experiments.runtime_workers import NativeScheduler

                tail = NativeScheduler(root)
                while True:
                    tail.tick()
                    if runtime.finish_if_drained(
                        root,
                        unfinished=tail.unfinished,
                        failed=any(j["status"] == "failed" for j in tail.jobs.values()),
                    ):
                        break
                    _publish(root, plan, state)
                    time.sleep(0.25)
                state.update(status=runtime.snapshot(root)["status"], stage="completed")
        except KeyboardInterrupt:
            state.update(status="stopped", stage="stopped")
            for file_state in state["files"].values():
                if file_state["status"] == "running":
                    file_state["status"] = "stopped"
        except BaseException as exc:
            state.update(
                status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}"
            )
            for file_state in state["files"].values():
                if file_state["status"] == "running":
                    file_state["status"] = "failed"
        runtime.publish(root, active=0, status=state["status"])
        _save(root, state)
        _publish(root, plan, state)
        return {
            "directory": str(root),
            "status": state["status"],
            "files": state["files"],
            "error": state.get("error"),
        }


def run(
    directory_plan,
    *,
    background=False,
    display_options=None,
    runtime_limit=None,
    keep_open=False,
):
    root = allocate(directory_plan)
    _, plan, state = load(root)
    _runtime_initialize(root, plan, state, limit=runtime_limit, keep_open=keep_open)
    if background:
        from smartsom.experiments.background import launch

        return launch(root, display_options=display_options)
    return execute_saved(root, display_options=display_options)
