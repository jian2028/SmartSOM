"""Read-only monitor for mainline runtime snapshots and historical manifests."""

import json
import math
import time
from pathlib import Path

from smartsom.telemetry.runtime import FINAL, SCHEMA, DisplayOptions, RuntimeDisplay

MAINLINE = {
    "smartsom.experiment/v2",
    "smartsom.evaluation/v1",
    "smartsom.production-run/v1",
    "smartsom.study-manifest/v1",
}


def _count(value):
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _read_snapshot(root):
    root = Path(root)
    path = root / "logs/progress.json"
    if path.is_file():
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("invalid runtime progress snapshot")
        if result.get("schema") != SCHEMA:
            raise ValueError("unsupported runtime progress schema")
        if (
            not isinstance(result.get("tasks"), list)
            or result.get("updated_at") is None
            or not _count(result.get("updated_at"))
            or not _count(result.get("total_tasks"))
        ):
            raise ValueError("invalid runtime progress snapshot")
        if not all(
            isinstance(result.get(key), str)
            for key in ("name", "kind", "stage", "status")
        ) or any(
            not isinstance(row, dict)
            or not all(
                isinstance(row.get(key), str)
                for key in ("id", "name", "stage", "status")
            )
            or not isinstance(row.get("values"), dict)
            or not isinstance(row.get("learner", {}), dict)
            or not _count(row.get("total"))
            or not _count(row.get("completed"))
            or not _count(row.get("stage_started_at"))
            for row in result["tasks"]
        ):
            raise ValueError("invalid runtime progress snapshot")
        overview = result.get("overview")
        if overview is not None and (
            not isinstance(overview, dict)
            or any(
                not _count(overview.get(key))
                for key in (
                    "work_completed",
                    "work_total",
                    "training_completed",
                    "training_total",
                    "elapsed_seconds",
                    "eta_seconds",
                )
            )
        ):
            raise ValueError("invalid runtime overview")
        from smartsom.telemetry.workflow import validate_workflow

        if result.get("workflow") is not None:
            validate_workflow(result["workflow"])
        for row in result["tasks"]:
            if row["values"].get("workflow") is not None:
                validate_workflow(row["values"]["workflow"])
        if result.get("tuning") is not None:
            from smartsom.telemetry.tuning_dashboard import clean_summary

            result["tuning"] = clean_summary(result["tuning"])
        return result
    manifest = root / "run.json"
    if not manifest.exists():
        manifest = root / "manifest.json"
    if not manifest.exists():
        raise ValueError(
            "no mainline run manifest or runtime progress snapshot; experimental formats are unsupported"
        )
    record = json.loads(manifest.read_text())
    if record.get("schema") not in MAINLINE:
        raise ValueError(
            "unsupported run format; W38, dispatch pilot and overnight are not supported"
        )
    status = record.get("status", "unknown")
    tasks = []
    events = root / "logs/events.jsonl"
    if events.is_file():
        # Read a bounded tail, tolerate a partially written last line.
        with events.open("rb") as stream:
            stream.seek(max(0, events.stat().st_size - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            from smartsom.telemetry.runtime import LABELS

            tasks.append(
                {
                    "id": str(root),
                    "name": record.get("name", root.name),
                    "status": status,
                    "stage": event.get("stage", status),
                    "values": {k: v for k, v in event.items() if k in LABELS},
                }
            )
            break
    return {
        "schema": SCHEMA,
        "name": record.get("name", root.name),
        "kind": record.get("kind", "historical"),
        "status": status,
        "stage": record.get("stage", status),
        "updated_at": manifest.stat().st_mtime,
        "tasks": tasks,
        "total_tasks": None,
        "notice": "Historical data: only recorded fields are available",
    }


def read_snapshot(root):
    result = _read_snapshot(root)
    from smartsom.experiments.control import ACTIVE, alive, processes, read, requested

    owner = read(root)
    if owner:
        state = owner["status"]
        if state in ACTIVE and requested(root):
            identities = [owner["owner"], *owner["members"]]
            table = processes()
            state = (
                "stopping"
                if any(alive(item, table) for item in identities)
                else "stopped"
            )
        if state in {"stopping", "stopped", "force_stopped"}:
            result.update(status=state, stage=state)
            result["notice"] = (
                "Control: " + state + "; recovery uses the latest committed checkpoint"
            )
    return result


def monitor(root, *, once=False, options=None, poll_seconds=1.0):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("monitor requires an existing run directory")
    first = read_snapshot(root)
    display = RuntimeDisplay(
        options or DisplayOptions(), kind=first["kind"], readonly=True
    )
    display.start()
    previous_state = None
    try:
        while True:
            try:
                snapshot = first if first is not None else read_snapshot(root)
                first = None
                display.from_snapshot(snapshot)
                age = max(0, time.time() - display.updated_at)
                if (
                    age > 5
                    and display.status not in FINAL
                    and display.status != "stopping"
                ):
                    display.notice = (
                        f"Last recorded update {age:.0f}s ago; process state unknown"
                    )
            except (OSError, ValueError, KeyError, TypeError):
                display.notice = (
                    "Snapshot temporarily unavailable; showing last valid update"
                )
            state = (display.stage, display.status)
            display.publish(force=once or state != previous_state)
            previous_state = state
            if once or display.status in FINAL:
                return 0
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        display.close()
