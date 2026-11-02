"""Local evidence serialization and source identity, outside the engine."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from smartsom._filesystem import atomic_replace, native_path
from smartsom.config.codec import canonical_json, digest, primitive

_RUNTIME_METADATA = ContextVar("runtime_package_metadata", default=None)


@contextmanager
def runtime_metadata():
    """One launch's installed-package inventory; source/git checks stay live."""
    if _RUNTIME_METADATA.get() is not None:
        yield
        return
    token = _RUNTIME_METADATA.set({})
    try:
        yield
    finally:
        _RUNTIME_METADATA.reset(token)


def implementation_identity():
    """Identity of executable sources, shared by every current execution path."""
    root = Path(__file__).resolve().parents[1]
    return digest(
        [
            (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.rglob("*.py"))
        ]
    )


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    native_path(temporary).write_text(
        json.dumps(
            primitive(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    atomic_replace(temporary, path)


def append_json(stream, value) -> None:
    stream.write(canonical_json(value) + "\n")
    stream.flush()


def source_identity() -> dict:
    # Find the checkout that supplies this code, not the caller's working directory.
    root = next(
        (p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None
    )
    git = {"commit": None, "dirty": None, "status": None}
    if root is not None:
        try:
            # One live Git observation supplies HEAD, branch and every status
            # record. Keep the complete machine-readable output as audit evidence.
            status = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain=v2",
                    "--branch",
                    "--untracked-files=all",
                ],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            ).strip()
            headers = {}
            records = []
            for line in status.splitlines():
                if line.startswith("# "):
                    key, _, value = line[2:].partition(" ")
                    headers[key] = value
                else:
                    records.append(line)
            commit = headers.get("branch.oid")
            branch = headers.get("branch.head")
            git = {
                "commit": None if commit == "(initial)" else commit,
                "branch": "" if branch == "(detached)" else branch,
                "dirty": bool(records),
                "status": status,
                "status_format": "porcelain-v2-branch",
            }
        except (OSError, subprocess.SubprocessError) as exc:
            git["unavailable_reason"] = str(exc)
    inventory = _RUNTIME_METADATA.get()
    versions = {} if inventory is None else inventory
    for package in (
        "smartsom",
        "pydantic",
        "pydantic-core",
        "PyYAML",
        "annotated-types",
        "typing-extensions",
        "typing-inspection",
        "pyjobshop",
        "ortools",
        "gymnasium",
        "numpy",
        "torch",
        "ray",
        "stable-baselines3",
        "sb3-contrib",
        "pettingzoo",
        "psutil",
        "threadpoolctl",
    ):
        if package in versions:
            continue
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git": git,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(versions),
    }


def runtime_source_matches(frozen: dict, live: dict) -> bool:
    """Match code/runtime identity while allowing unrelated working-tree edits.

    The complete source snapshot remains in the manifest. The separate
    implementation digest and frozen inputs guard executable/scientific changes.
    """
    return (
        live.get("python") == frozen.get("python")
        and live.get("packages") == frozen.get("packages")
        and live.get("git", {}).get("commit") == frozen.get("git", {}).get("commit")
    )


def artifact_digests(run_dir: Path) -> dict[str, str]:
    # Manifest excludes itself, avoiding a recursive self-checksum.
    return {
        path.name: _file_digest(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
        and path.name != "manifest.json"
        and not path.name.endswith(".tmp")
    }


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
