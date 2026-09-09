"""Describe observed input overlap without treating new seeds as new scenarios."""

import hashlib
import json
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments.training_audit import load_training_snapshot


def input_coverage(snapshot, entries):
    """Compare saved input values; unavailable training history stays unavailable.

    Checkpoint ledgers cover the recorded episodes at that save point. They need
    not include an active episode, so absence of overlap is never a disjointness
    claim about all samples used by the model.
    """
    worlds = {row["world_sha256"] for _, row in entries}
    result = {
        "schema": "smartsom.input-coverage/v1",
        "evaluation_inputs": len(
            {(r["case_id"], r["replication"]) for _, r in entries}
        ),
        "evaluation_unique_worlds": len(worlds),
        "training_history": {"status": "unavailable"},
        "validation": {"status": "unavailable"},
        "cases": [],
        "interpretation": "Different seeds alone do not establish cross-scenario generalization.",
    }
    if snapshot is None:
        return result
    snapshot = Path(snapshot)
    resolved = load_training_snapshot(snapshot)
    for case in sorted({row["case_id"] for _, row in entries}):
        rows = [row for _, row in entries if row["case_id"] == case]
        result["cases"].append(
            {
                "case_id": case,
                "same_training_factory": all(
                    r["input_sha256"]["factory"] == resolved.base.factory_sha256
                    for r in rows
                ),
                "same_training_workload": all(
                    r["input_sha256"]["workload"] == resolved.base.workload_sha256
                    for r in rows
                ),
            }
        )
    ledger = snapshot.parent / "episodes.jsonl"
    if ledger.is_file():
        rows = [
            json.loads(line) for line in ledger.read_text().splitlines() if line.strip()
        ]
        observed = {row["input_sha256"] for row in rows}
        result["training_history"] = {
            "status": "recorded_episodes_only",
            "source": str(ledger),
            "source_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "episodes": len(rows),
            "unique_worlds": len(observed),
            "overlapping_evaluation_worlds": sorted(worlds & observed),
            "all_training_samples_covered": False,
        }
    # The nearest saved input set belongs to this checkpoint's training attempt.
    for parent in (snapshot.parent, *tuple(snapshot.parent.parents)[:2]):
        path = parent / "validation_inputs.json"
        if path.is_file():
            validation = json.loads(path.read_text())
            values = {digest(item["episode"]) for item in validation}
            result["validation"] = {
                "status": "available",
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "inputs": len(validation),
                "unique_worlds": len(values),
                "overlapping_evaluation_worlds": sorted(worlds & values),
                "worlds_disjoint": not bool(worlds & values),
            }
            break
    return result
