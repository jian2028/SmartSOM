"""Verify and protect historical checkpoint bytes; no legacy training executor.

New checkpoints and exact resume use production_training. These helpers only
verify old artifacts and register retention references for historical reports.
"""

import json
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments.evidence import write_json
from smartsom.learning.checkpoint import file_hash

SCHEMA = "smartsom.update-checkpoint/v1"


def _members(directory):
    return {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p != directory / "manifest.json"
        and not p.relative_to(directory).parts[0] == "references"
    }


def inspect_resume_checkpoint(path):
    """Verify all members before deserializing a local trusted framework artifact."""
    directory = Path(path).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("not a complete resumable update checkpoint")
    if manifest["files"] != _members(directory):
        raise ValueError("resumable checkpoint member digests disagree")
    return manifest


def protect_checkpoint(path, reference):
    """Register an explicit report/reference so automatic retention cannot delete it."""
    inspect_resume_checkpoint(path)
    directory = Path(path) / "references"
    directory.mkdir(exist_ok=True)
    write_json(
        directory / f"{digest(str(reference))}.json", {"reference": str(reference)}
    )
