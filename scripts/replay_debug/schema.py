"""Trace schema ``smartsom.somarl.replay.v1``: write, load and validate.

See ``README.md`` for the field-by-field description. Validation returns a
list of human-readable problems instead of raising, so a partially broken
trace from a new simulator can still be inspected in the viewer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "smartsom.somarl.replay.v1"
MANIFEST_FILE = "manifest.json"
REPLAY_FILE = "replay.jsonl"

AGENT_TYPES = ("machine", "buffer", "quality", "dispatcher", "mover")
AGV_STATUSES = ("idle", "moving", "servicing", "waiting", "charging")
MACHINE_STATUSES = ("idle", "processing", "blocked", "breakdown")
QUALITY_STATES = ("unknown", "pass", "fail")

# Event type -> required fields (beyond "type").
EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "job_released": ("job",),
    "dispatch": ("agv", "kind", "station"),
    "pickup": ("agv", "job", "station"),
    "dropoff": ("agv", "job", "station"),
    "service_conflict": ("agv", "station", "reason"),
    "movement_conflict": ("agv", "kind", "cell"),
    "op_start": ("machine", "job", "mode"),
    "op_done": ("machine", "job", "defect_introduced"),
    "machine_blocked": ("machine", "job"),
    "inspection_start": ("station", "jobs"),
    "inspection_done": ("station", "results"),
    "delivered": ("job", "observed_quality", "defective", "lateness", "on_time"),
    "scrapped": ("job", "station"),
}

MANIFEST_FIELDS = (
    "schema_version",
    "trace_id",
    "controller",
    "seed",
    "horizon",
    "layout",
    "agents",
    "jobs",
)
FRAME_FIELDS = (
    "tick",
    "agvs",
    "machines",
    "buffers",
    "inspection",
    "jobs",
    "decisions",
    "rewards",
    "events",
)


def new_manifest(**fields: Any) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(fields)
    return manifest


def write_trace(
    trace_dir: Path | str, manifest: dict[str, Any], frames: Iterable[dict[str, Any]]
) -> Path:
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (trace_dir / REPLAY_FILE).open("w", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame, separators=(",", ":")) + "\n")
    return trace_dir


def load_trace(path: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a trace directory, or a single bundled ``{"manifest", "frames"}`` JSON file."""
    path = Path(path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["manifest"], payload["frames"]
    manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    frames = []
    with (path / REPLAY_FILE).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                frames.append(json.loads(line))
    return manifest, frames


def validate_trace(
    manifest: dict[str, Any], frames: list[dict[str, Any]], max_errors: int = 50
) -> list[str]:
    errors: list[str] = []

    def err(msg: str) -> None:
        if len(errors) < max_errors:
            errors.append(msg)

    for key in MANIFEST_FIELDS:
        if key not in manifest:
            err(f"manifest: missing '{key}'")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        err(
            f"manifest: schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    if errors:
        return errors

    layout = manifest["layout"]
    obstacles = {tuple(c) for c in layout.get("obstacles", [])}
    width, height = int(layout["width"]), int(layout["height"])
    station_ids = {s["id"] for s in layout.get("stations", [])}
    for agent in manifest["agents"]:
        if agent.get("type") not in AGENT_TYPES:
            err(
                f"manifest: agent {agent.get('id')!r} has unknown type {agent.get('type')!r}"
            )
    known_jobs = {j["id"] for j in manifest["jobs"]}

    if not frames:
        err("replay: no frames")
        return errors

    prev_tick = None
    for i, frame in enumerate(frames):
        where = f"frame[{i}]"
        missing = [k for k in FRAME_FIELDS if k not in frame]
        if missing:
            err(f"{where}: missing {missing}")
            continue
        tick = frame["tick"]
        where = f"tick {tick}"
        if prev_tick is not None and tick != prev_tick + 1:
            err(f"{where}: ticks must increase by 1 (previous {prev_tick})")
        prev_tick = tick

        seen_cells: dict[tuple[int, int], str] = {}
        for agv in frame["agvs"]:
            cell = tuple(agv["pos"])
            if (
                not (0 <= cell[0] < width and 0 <= cell[1] < height)
                or cell in obstacles
            ):
                err(f"{where}: {agv['id']} at {list(cell)} is not a road cell")
            if cell in seen_cells:
                err(
                    f"{where}: {agv['id']} and {seen_cells[cell]} share cell {list(cell)}"
                )
            seen_cells[cell] = agv["id"]
            if agv.get("status") not in AGV_STATUSES:
                err(f"{where}: {agv['id']} has unknown status {agv.get('status')!r}")

        for machine in frame["machines"]:
            if machine.get("status") not in MACHINE_STATUSES:
                err(
                    f"{where}: {machine['id']} has unknown status {machine.get('status')!r}"
                )

        for buf in frame["buffers"]:
            if buf["id"] not in station_ids:
                err(f"{where}: unknown buffer {buf['id']!r}")
            if (
                buf.get("nominated") is not None
                and buf["nominated"] not in buf["slots"]
            ):
                err(
                    f"{where}: {buf['id']} nominates {buf['nominated']!r} which is not in its slots"
                )

        for jid, job in frame["jobs"].items():
            if jid not in known_jobs:
                err(f"{where}: job {jid!r} not declared in manifest.jobs")
            if job.get("quality") not in QUALITY_STATES:
                err(f"{where}: job {jid!r} has unknown quality {job.get('quality')!r}")

        for event in frame["events"]:
            etype = event.get("type")
            required = EVENT_FIELDS.get(etype)
            if required is None:
                err(f"{where}: unknown event type {etype!r}")
                continue
            absent = [k for k in required if k not in event]
            if absent:
                err(f"{where}: {etype} event missing {absent}")

    return errors
