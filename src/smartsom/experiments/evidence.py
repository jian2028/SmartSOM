"""Local evidence serialization and source identity, outside the engine."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from smartsom.config.codec import canonical_json, primitive


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
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
    temporary.replace(path)


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

            def command(*args):
                return subprocess.check_output(
                    ["git", "-C", str(root), *args],
                    text=True,
                    stderr=subprocess.PIPE,
                    timeout=5,
                ).strip()

            commit = command("rev-parse", "HEAD")
            status = command("status", "--porcelain", "--untracked-files=all")
            git = {"commit": commit, "dirty": bool(status), "status": status}
        except (OSError, subprocess.SubprocessError) as exc:
            git["unavailable_reason"] = str(exc)
    versions = {}
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
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git": git,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


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
