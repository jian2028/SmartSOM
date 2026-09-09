"""Read local experiment evidence and regenerate external navigation views.

The catalog is a cache, never an experiment's authority. No function in this
module writes into a discovered run, including historical v1 evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from smartsom.experiments.evidence import write_json

EXPERIMENT_SCHEMA = "smartsom.experiment/v2"
CURRENT_SCHEMAS = {EXPERIMENT_SCHEMA, "smartsom.evaluation/v1"}
LEGACY_SCHEMAS = {
    "smartsom.training-manifest/v1": "train",
    "smartsom.study-manifest/v1": "study",
    "smartsom.manifest/v1": "evaluate",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def contained_path(root: Path, value: str) -> Path:
    """Resolve a declared relative path, rejecting absolute/escaping references."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"expected a nonempty relative artifact path: {value!r}")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes experiment directory: {value!r}")
    return candidate


@dataclass(frozen=True)
class RunEntry:
    id: str
    name: str
    kind: str
    status: str
    path: Path
    provider: str | None = None
    seed: int | None = None
    tags: tuple[str, ...] = ()


def read_run(source: str | Path) -> RunEntry:
    root = Path(source).resolve()
    current = root / "run.json"
    if current.is_file() and read_json(current).get("schema") in CURRENT_SCHEMAS:
        data = read_json(current)
        if data["schema"] == "smartsom.evaluation/v1":
            data.setdefault("name", root.name)
        for key in ("id", "name", "kind", "status"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"invalid experiment {key}: {current}")
        paths = data.get("paths", {})
        if not isinstance(paths, dict) or any(not isinstance(k, str) for k in paths):
            raise ValueError(f"invalid experiment paths: {current}")
        for value in paths.values():
            contained_path(root, value)
        return RunEntry(
            *(data[key] for key in ("id", "name", "kind", "status")),
            path=root,
            provider=data.get("provider"),
            seed=data.get("seed"),
            tags=tuple(data.get("tags", ())),
        )
    manifest = root / "manifest.json"
    data = read_json(manifest)
    if data.get("schema") not in LEGACY_SCHEMAS:
        raise ValueError(f"unrecognized run manifest: {manifest}")
    return RunEntry(
        id=root.name,
        name=root.name,
        kind=LEGACY_SCHEMAS[data["schema"]],
        status=data.get("status", "unknown"),
        path=root,
        provider=data.get("provider"),
    )


def _roots(roots: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    if isinstance(roots, (str, Path)):
        roots = (roots,)
    return tuple(dict.fromkeys(Path(root).resolve() for root in roots))


def list_runs(roots: str | Path | Iterable[str | Path]) -> tuple[RunEntry, ...]:
    """Discover v2 experiments and legacy runs/studies without traversing links.

    A recognized experiment owns its children, so it appears once in the catalog.
    An explicit legacy training or study directory can also be supplied as a root.
    """
    found = {}
    for root in _roots(roots):
        if not root.is_dir():
            continue
        for directory, names, files in os.walk(root, followlinks=False):
            names[:] = sorted(
                n for n in names if not n.startswith(".") and n != "__pycache__"
            )
            path = Path(directory)
            candidate = None
            if "run.json" in files:
                if read_json(path / "run.json").get("schema") in CURRENT_SCHEMAS:
                    candidate = read_run(path)
            if candidate is None and "manifest.json" in files:
                if read_json(path / "manifest.json").get("schema") in LEGACY_SCHEMAS:
                    candidate = read_run(path)
            if candidate is not None:
                found[path] = candidate
                names.clear()
    return tuple(
        sorted(found.values(), key=lambda r: (r.id, str(r.path)), reverse=True)
    )


def training_locator(source: str | Path) -> Path:
    """Find the authoritative training evidence in a current or historical run."""
    root = Path(source).resolve()
    current = root / "run.json"
    if current.is_file() and read_json(current).get("schema") == EXPERIMENT_SCHEMA:
        value = read_json(current).get("paths", {}).get("training")
        if value is None:
            raise ValueError("experiment has no training evidence")
        root = contained_path(root, value)
    if not (root / "resolved_training.json").is_file():
        raise ValueError(f"training snapshot is unavailable in {root}")
    return root


def resolve_run(query: str | Path, roots: str | Path | Iterable[str | Path]) -> Path:
    """Resolve an exact path/name/id or unique id prefix; never pick 'latest'."""
    direct = Path(query)
    if direct.is_dir():
        return read_run(direct).path
    entries = list_runs(roots)
    exact = [r for r in entries if str(query) in (r.id, r.name)]
    candidates = exact or [r for r in entries if r.id.startswith(str(query))]
    if len(candidates) != 1:
        detail = ", ".join(str(r.path) for r in candidates)
        raise ValueError(
            f"{'ambiguous' if candidates else 'unknown'} run {str(query)!r}"
            + (f": {detail}" if detail else "")
        )
    return candidates[0].path


def artifact_paths(source: str | Path) -> dict[str, Path]:
    """Return existing useful artifacts for one current or historical run."""
    root = Path(source).resolve()
    read_run(root)
    result = {}
    current = root / "run.json"
    if current.is_file() and read_json(current).get("schema") in CURRENT_SCHEMAS:
        result = {
            key: contained_path(root, value)
            for key, value in read_json(current).get("paths", {}).items()
            if contained_path(root, value).exists()
        }
    training = result.get("training", root)
    if "checkpoint" not in result and (training / "checkpoints/last.json").is_file():
        from smartsom.experiments.packaging import model_locator

        result["checkpoint"] = model_locator(training)
    if (
        "checkpoint" not in result
        and (training / "checkpoint_algorithm.json").is_file()
    ):
        from smartsom.experiments.packaging import locate_reference

        descriptor = training / "checkpoint_algorithm.json"
        algorithm = read_json(descriptor)["algorithm"]
        checkpoint = locate_reference(descriptor, algorithm["checkpoint"])
        if checkpoint.is_dir():
            result["checkpoint"] = checkpoint
    if "logs" not in result:
        result["logs"] = root / "logs" if (root / "logs").is_dir() else training
    if "report" not in result:
        for candidate in (
            root / "reports",
            root / "report.html",
            root / "summary.json",
        ):
            if candidate.exists():
                result["report"] = candidate
                break
    return result


def _link_name(entry: RunEntry) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", entry.name).strip(".-")[:64] or "run"
    identity = hashlib.sha256(str(entry.path).encode()).hexdigest()[:12]
    return f"{label}-{identity}"


def rebuild_views(
    roots: str | Path | Iterable[str | Path], views_dir: str | Path
) -> Path:
    """Rebuild index.json and relative models/logs/reports links outside runs.

    Only symlinks recorded by the previous index are removed, and only if their
    current target still matches. Ordinary files and user-modified links are kept.
    """
    entries = list_runs(roots)
    destination = Path(views_dir).resolve()
    if any(destination.is_relative_to(entry.path) for entry in entries):
        raise ValueError("catalog views must be outside every experiment directory")
    destination.mkdir(parents=True, exist_ok=True)
    for group in ("models", "logs", "reports"):
        if (destination / group).is_symlink():
            raise ValueError(f"catalog group cannot be a symlink: {group}")
    index = destination / "index.json"
    previous = read_json(index) if index.exists() else {}
    desired = {}
    records = []
    for entry in entries:
        record = asdict(entry)
        record["path"] = os.path.relpath(entry.path, destination)
        records.append(record)
        paths = artifact_paths(entry.path)
        for group, key in (
            ("models", "checkpoint"),
            ("logs", "logs"),
            ("reports", "report"),
        ):
            if key in paths:
                link = Path(group) / _link_name(entry)
                desired[link.as_posix()] = os.path.relpath(
                    paths[key], destination / group
                )
    for name, target in previous.get("links", {}).items():
        relative = Path(name)
        if relative.parent.as_posix() not in {"models", "logs", "reports"}:
            raise ValueError("invalid catalog link entry")
        link = destination / relative
        if name not in desired and link.is_symlink() and os.readlink(link) == target:
            link.unlink()
    for name, target in desired.items():
        link = destination / name
        link.parent.mkdir(exist_ok=True)
        if link.is_symlink():
            if os.readlink(link) == target:
                continue
            if previous.get("links", {}).get(name) != os.readlink(link):
                raise FileExistsError(f"refusing to replace user link: {link}")
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"refusing to replace existing file: {link}")
        link.symlink_to(target, target_is_directory=(link.parent / target).is_dir())
    write_json(
        index, {"schema": "smartsom.catalog/v1", "runs": records, "links": desired}
    )
    return index
