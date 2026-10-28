"""Read-only run discovery, with stable identities independent of directory moves."""

import hashlib
import json
import shlex
from pathlib import Path

from smartsom.config.launch import project_root


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def describe(root):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    manifest = read_json(root / "manifest.json")
    run = read_json(root / "run.json")
    progress = (
        read_json(root / "progress.json") or read_json(root / "batch.json") or run
    )
    execution = read_json(root / "execution.json")
    tracking = read_json(root / "tracking/status.json")
    current = plan.get(
        "runtime_contract"
    ) == "smartsom.current-runtime/v1" and plan.get("schema") in {
        "smartsom.reservation-plan/v1",
        "smartsom.author-batch-plan/v1",
    }
    identity = (
        plan.get("run_id") or manifest.get("run_id") or run.get("id") or str(root)
    )
    short = hashlib.sha256(str(identity).encode()).hexdigest()[:12]
    authoring = read_json(root / "authoring.json")
    from smartsom.config.codec import read_document

    inputs = root / "inputs"
    algorithm = (
        read_document(inputs / "algorithm.yaml")
        if (inputs / "algorithm.yaml").is_file()
        else {}
    )
    experiment = (
        read_document(inputs / "experiment.yaml")
        if (inputs / "experiment.yaml").is_file()
        else {}
    )
    replay = [p.parent for p in sorted(root.glob("replay-*/run.json"))]
    if run.get("schema") in {
        "smartsom.production-run/v1",
        "smartsom.production-run/v2",
    }:
        replay.append(root)
    commands = [shlex.join(["smartsom", "playback", str(path)]) for path in replay]
    if run.get("schema") == "smartsom.spatial-run/v1":
        commands.extend(
            shlex.join(
                [
                    "smartsom",
                    "playback",
                    str(root),
                    "--seed",
                    p.stem.removeprefix("frames-"),
                ]
            )
            for p in sorted(root.glob("frames-*.jsonl"))
        )
    return {
        "id": str(identity),
        "short_id": short,
        "directory": str(root),
        "status": progress.get("status", "unknown"),
        "current": current,
        "algorithm": algorithm.get("method"),
        "seed": manifest.get("seed", experiment.get("run", {}).get("seed")),
        "scientific_sha256": plan.get(
            "scientific_sha256", authoring.get("scientific_sha256")
        ),
        "source": manifest.get("source", plan.get("source", run.get("source"))),
        "tracking": tracking,
        "operations": execution,
        "progress": progress,
        "summary": read_json(root / "summary.json"),
        "replay_directories": list(map(str, replay)),
        "replay_commands": commands,
        "recovery": "current frozen runtime"
        if current
        else (
            "resume through parent Tune run"
            if manifest.get("runtime_contract") == "smartsom.current-runtime/v1"
            else "archived source required"
        ),
    }


def list_runs(roots=None):
    roots = roots or [project_root() / "runs"]
    directories = set()
    for source in roots:
        root = Path(source).expanduser().resolve()
        for name in ("plan.json", "run.json", "manifest.json"):
            for path in root.rglob(name):
                # Exclude frozen source, checkpoints and tracking SDK internals.
                relative = path.relative_to(root)
                if not set(relative.parts) & {
                    "inputs",
                    "source",
                    "tracking",
                    "continuation",
                    "checkpoints",
                    "replay",
                }:
                    directories.add(path.parent)
    rows = []
    for directory in sorted(directories):
        try:
            rows.append(describe(directory))
        except (ValueError, OSError) as error:
            rows.append(
                {
                    "directory": str(directory),
                    "status": "unreadable",
                    "error": str(error),
                }
            )
    return rows


def resolve_run(value, roots=None):
    path = Path(value).expanduser()
    if path.is_dir() and any(
        (path / name).is_file() for name in ("plan.json", "run.json", "manifest.json")
    ):
        return path.resolve()
    matches = [
        row
        for row in list_runs(roots)
        if any(str(row.get(k, "")).startswith(str(value)) for k in ("id", "short_id"))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"run ID {value!r}: expected one match; candidates: "
            + ", ".join(row["directory"] for row in matches)
        )
    return Path(matches[0]["directory"])


def require_current(root):
    if not describe(root)["current"]:
        raise ValueError(
            "historical runs are read-only; use archived source: docs/archive/index.json"
        )


def results(root, *, algorithm=None, seed=None, status=None):
    rows = list_runs([root])
    return [
        row
        for row in rows
        if all(
            value is None or row.get(key) == value
            for key, value in (
                ("algorithm", algorithm),
                ("seed", seed),
                ("status", status),
            )
        )
    ]
