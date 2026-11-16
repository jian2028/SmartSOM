"""Frozen current experiment directories, scheduled by official Ray Tune."""

import json
from dataclasses import dataclass
from pathlib import Path

from smartsom.config.codec import ConfigurationError, digest
from smartsom.experiments.evidence import (
    implementation_identity,
    runtime_source_matches,
    source_identity,
    write_json,
)
from smartsom.telemetry.runtime import CURRENT, operation

SCHEMA = "smartsom.author-batch-plan/v1"
STATE_SCHEMA = "smartsom.author-batch-state/v1"
RUN_SCHEMA = "smartsom.author-batch-run/v1"
FINAL = {"completed", "failed", "stopped"}


@dataclass(frozen=True)
class DirectoryPlan:
    directory: Path
    files: tuple
    output_root: Path
    selection: dict | None = None
    contract: str = "reservation"
    operations: dict | None = None
    research_layout: bool = False
    incremental: bool = False
    resume_missing: bool = False
    inventory: dict | None = None

    def summary(self):
        return {
            "status": "checked",
            "input_type": "reservation-directory",
            "directory": str(self.directory),
            "output_root": str(self.output_root),
            "count_files": len(self.files),
            "execution_selection": self.selection,
            "files": [
                {
                    "id": row["id"],
                    "stage": row["stage"],
                    **row["compiled"].summary(),
                }
                for row in self.files
            ],
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
    selectors=None,
    first_train_seeds=None,
    resume_missing=False,
    **overrides,
):
    """Check every direct YAML member, failing on an unsupported or duplicate input."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigurationError(f"run requires an Experiment directory: {directory}")
    if any(
        value is not None
        for value in (calibration_seconds, calibration_level, calibration_candidate)
    ):
        raise ConfigurationError(
            "performance calibration was removed; declare trial_resources in the profile"
        )
    from smartsom.config.experiment_layout import entry_id, entrypoints

    paths = entrypoints(directory)
    if not paths:
        raise ConfigurationError("directory contains no Experiment YAML entries")
    from smartsom.experiments.batch_selection import requests

    selection_requests = requests(ids, selectors, first_train_seeds, overrides)
    incremental = (
        first_train_seeds is not None
        or resume_missing
        or ids is not None
        or any(path.name != "experiment.yaml" for path in paths)
    )
    if first_train_seeds is not None and (
        type(first_train_seeds) is not int or first_train_seeds < 1
    ):
        raise ConfigurationError("--first-train-seeds must be a positive integer")
    all_paths = paths
    inventory = {entry_id(p, directory): p for p in paths}
    if len(inventory) != len(paths):
        raise ConfigurationError("duplicate batch Experiment IDs")
    if ids is not None:
        if not ids:
            raise ConfigurationError("batch IDs must be nonempty")
        unknown = set(ids) - set(inventory)
        if unknown:
            raise ConfigurationError(
                "unknown batch Experiment IDs: " + ", ".join(sorted(unknown))
            )
        paths = [inventory[identity] for identity in dict.fromkeys(ids)]
    if tags is not None:
        if not tags or any(not tag for tag in tags) or len(set(tags)) != len(tags):
            raise ConfigurationError("batch tags must be nonempty and unique")
        from smartsom.config.codec import read_document

        matched = []
        for path in paths:
            document = read_document(path)
            output = document.get("run", {})
            if not isinstance(output, dict):
                raise ConfigurationError(f"{path.name}: output must be a mapping")
            authored = output.get("tags", [])
            if not isinstance(authored, list) or any(
                type(tag) is not str for tag in authored
            ):
                raise ConfigurationError(
                    f"{path.name}: run.tags must be a list of strings"
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
    if ids is None:
        selection_requests = [
            {"entry": entry_id(path, directory), "selector": None} for path in paths
        ]
    from smartsom.config.reservation import is_reservation

    reservation_inputs = [is_reservation(path) for path in paths]
    if any(reservation_inputs):
        if not all(reservation_inputs):
            raise ConfigurationError(
                "mixing Base and V4 contracts in one batch is unsupported"
            )
        from smartsom.experiments.reservation_batch import compile_files

        return compile_files(
            directory,
            all_paths if incremental else paths,
            execution_selection={
                "entries": [entry_id(p, directory) for p in paths],
                "first_train_seeds": first_train_seeds,
                "requests": selection_requests,
            }
            if incremental
            else None,
            resume_missing=resume_missing,
            **overrides,
        )
    raise ConfigurationError(
        "historical batch inputs require archived source; see docs/archive/index.json"
    )


def allocate(directory_plan):
    from smartsom.experiments.reservation_batch import allocate_directory

    return allocate_directory(directory_plan)


def load(root):
    from smartsom.experiments.reservation import load as load_reservation

    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if plan.get("contract") != "reservation":
        raise ValueError("historical batch recovery requires archived source")
    if (
        plan.get("schema") != SCHEMA
        or state.get("schema") != STATE_SCHEMA
        or digest(plan) != state.get("plan_sha256")
    ):
        raise ValueError("frozen directory batch changed")
    if (
        not runtime_source_matches(plan["source"], source_identity())
        or implementation_identity() != plan["implementation_sha256"]
    ):
        raise ValueError("batch source/dependency identity changed; create a new run")
    for row in plan["files"]:
        child = (root / row["child"]).resolve()
        if not child.is_relative_to(root) or child == root:
            raise ValueError("batch child escapes its run")
        child_plan = load_reservation(child)
        if digest(child_plan) != row["child_plan_sha256"]:
            raise ValueError("frozen child plan changed")
    return root, plan, state


def _save(root, state):
    from smartsom.experiments.incremental_batch import counts

    state.update(counts(state))
    if state.get("stage") == "selection_completed":
        from smartsom.experiments.incremental_batch import completion

        state["status"] = completion(state)
    write_json(root / "batch.json", state)
    record = json.loads((root / "run.json").read_text())
    record.update(status=state["status"], stage=state["stage"], files=state["files"])
    write_json(root / "run.json", record)


def _publish(root, plan, state):
    from smartsom.telemetry.current_progress import presentation

    session = CURRENT.get()
    if session is None:
        return
    session.configure_batch(
        [
            dict(id=row["id"], stage=row["stage"], barrier_group=str(root), requires=[])
            for row in plan["files"]
        ]
    )
    session.phase(state["stage"])
    with session.batch_updates():
        for row in plan["files"]:
            status = state["files"][row["id"]]["status"]
            progress_path = root / row["child"] / "progress.json"
            progress = (
                json.loads(progress_path.read_text()) if progress_path.exists() else {}
            )
            details, total, unit = presentation(progress, task=row["task"])
            session.update(
                row["id"],
                {
                    **details,
                    "display_name": row["id"],
                    "status": status,
                    "stage": details["stage"] if progress else status,
                    "scheduler_status": status,
                    "context": f"stage {row['stage']} · {row['task']}",
                },
                total=total,
                unit=unit,
                final=status in FINAL,
            )


def _gate(root, row):
    path = root / row["child"] / "cases.jsonl"
    cases = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    if any(case.get("engineering_failure") for case in cases):
        raise RuntimeError(f"{row['id']}: engineering failure in evaluation")
    return {"cases": len(cases), "passed": True}


@operation("batch-directory")
def execute_saved(root, *, retry_failed=False):
    from smartsom.experiments.tune_batch import execute_directory

    return execute_directory(root, retry_failed=retry_failed)


def run(
    directory_plan,
    *,
    background=False,
    display_options=None,
    keep_open=False,
):
    if keep_open:
        raise ValueError("online additions were removed; launch a frozen Tune cohort")
    root = allocate(directory_plan)
    if background:
        from smartsom.experiments.background import launch

        return launch(root, display_options=display_options)
    return execute_saved(root, display_options=display_options)
