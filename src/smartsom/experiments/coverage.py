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


def grid_input_coverage(snapshot, entries, worlds_by_case):
    """Compare frozen grid inputs and retained histories without rerunning generation."""
    from smartsom.config.production import ProductionRecipe

    worlds = {row["world_sha256"] for row in entries}
    result = {
        "schema": "smartsom.input-coverage/v1",
        "evaluation_inputs": len(worlds_by_case),
        "evaluation_unique_worlds": len(worlds),
        "training_history": {"status": "unavailable"},
        "validation": {"status": "unavailable"},
        "cases": [],
        "interpretation": "Different seeds alone do not establish cross-scenario generalization.",
    }
    if snapshot is None:
        return result
    snapshot = Path(snapshot)
    recipe = ProductionRecipe(**json.loads(snapshot.read_text())["recipe"])

    def jobs(case):
        # Arrival timing is a separate disturbance, not a new product route.
        return digest(
            [
                (d.demand_id, d.steps, d.due_at, d.priority, d.input_id)
                for d in sorted(case.demands, key=lambda d: d.demand_id)
            ]
        )

    for case_id in sorted({case for case, _ in worlds_by_case}):
        cases = [case for (key, _), case in worlds_by_case.items() if key == case_id]
        result["cases"].append(
            {
                "case_id": case_id,
                "same_training_factory": all(
                    digest(case.factory) == digest(recipe.scenario.factory)
                    for case in cases
                ),
                "same_training_workload": all(
                    jobs(case) == jobs(recipe.scenario) for case in cases
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
    validation_path = snapshot.parent / "validation.json"
    if validation_path.is_file():
        report = json.loads(validation_path.read_text())
        rows = report["results"]
        if all("world_sha256" in row for row in rows):
            values = {row["world_sha256"] for row in rows}
            result["validation"] = {
                "status": "available",
                "source": str(validation_path),
                "source_sha256": hashlib.sha256(
                    validation_path.read_bytes()
                ).hexdigest(),
                "inputs": len(rows),
                "unique_worlds": len(values),
                "overlapping_evaluation_worlds": sorted(worlds & values),
                "worlds_disjoint": not bool(worlds & values),
            }
    return result
