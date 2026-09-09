"""Read evaluation coverage and pair identities without pooling training seeds."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean

from smartsom.experiments.catalog import read_json
from smartsom.experiments.evidence import _file_digest
from smartsom.experiments.packaging import locate_reference


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _identity(owner, row, warnings, models):
    checkpoint = row.get("checkpoint") or {}
    seed = None
    if checkpoint.get("path"):
        try:
            path = locate_reference(owner, checkpoint["path"])
            if not (path / "checkpoint.json").is_file():
                raise ValueError("checkpoint manifest is unavailable")
            expected = checkpoint.get("manifest_sha256")
            if expected and _file_digest(path / "checkpoint.json") != expected:
                raise ValueError("checkpoint manifest digest mismatch")
            models.add(str(path))
        except (OSError, ValueError) as exc:
            warnings.add(f"{row.get('algorithm_id', 'model')}: {exc}")
    if checkpoint.get("training_snapshot"):
        try:
            snapshot = locate_reference(owner, checkpoint["training_snapshot"])
            expected = checkpoint.get("training_snapshot_sha256")
            if expected and _file_digest(snapshot) != expected:
                raise ValueError("training snapshot digest mismatch")
            data = read_json(snapshot)
            if data.get("schema") != "smartsom.resolved-training/v1":
                raise ValueError("unknown training snapshot schema")
            seed = data["resolved"]["run"]["seed"]
            if type(seed) is not int:
                raise ValueError("training seed is not an integer")
        except (OSError, ValueError, KeyError) as exc:
            warnings.add(f"{row.get('algorithm_id', 'model')}: {exc}")
            seed = None
    return (
        row["case_id"],
        row["algorithm_id"],
        checkpoint.get("manifest_sha256"),
        seed,
        checkpoint.get("training_snapshot_sha256")
        or checkpoint.get("training_snapshot"),
    )


def evaluation_data(directory, root):
    """Keep each evaluation, model digest and training seed as a separate group."""
    owner = directory / "run.json"
    record = read_json(owner)
    plan_path = directory / "plan.json"
    planned = read_json(plan_path).get("entries", []) if plan_path.is_file() else []
    observed = record.get("results", [])
    warnings, models = set(), set()
    expected, actual = defaultdict(list), defaultdict(list)
    identities = {}
    for is_plan, rows, target in ((True, planned, expected), (False, observed, actual)):
        for row in rows:
            identity = _identity(plan_path if is_plan else owner, row, warnings, models)
            identities[id(row)] = identity
            target[identity].append(row)
    coverage = []
    for key in sorted(set(expected) | set(actual), key=str):
        rows = actual[key]
        complete = [
            r
            for r in rows
            if r.get("status") == "completed" and _number(r.get("makespan"))
        ]
        wanted_slots = Counter(r["replication"] for r in expected[key])
        seen_slots = Counter(r["replication"] for r in rows)
        missing = sum((wanted_slots - seen_slots).values())
        duplicates = sum(max(0, count - 1) for count in seen_slots.values())
        reasons = Counter(
            r.get("reason") or r.get("status", "unknown")
            for r in rows
            if r not in complete
        )
        if missing:
            reasons["missing_result"] += missing
        coverage.append(
            {
                "case_id": key[0],
                "algorithm_id": key[1],
                "checkpoint_sha256": key[2],
                "training_seed": key[3],
                "training_snapshot_identity": key[4],
                "requested": len(expected[key]) if plan_path.is_file() else None,
                "observed": len(rows),
                "completed": len(complete),
                "missing": missing,
                "duplicates": duplicates,
                "failure_reasons": dict(reasons),
                "complete_case_mean_makespan": mean(r["makespan"] for r in complete)
                if complete and not duplicates
                else None,
            }
        )
    pairs = []
    slots = defaultdict(lambda: defaultdict(list))
    plan_slots = defaultdict(lambda: defaultdict(list))
    for rows, target in ((observed, slots), (planned, plan_slots)):
        for row in rows:
            target[(row["case_id"], row["replication"])][row["algorithm_id"]].append(
                row
            )
    for case, replication in sorted(set(slots) | set(plan_slots)):
        rows, wanted = slots[case, replication], plan_slots[case, replication]
        for baseline in sorted((set(rows) | set(wanted)) - {"model"}):
            left, right = rows["model"], rows[baseline]
            status, delta = "paired", None
            if len(left) > 1 or len(right) > 1:
                status = "duplicate_result"
            elif not left or not right:
                status = "missing_model" if not left else "missing_baseline"
            elif not plan_path.is_file() or not wanted["model"] or not wanted[baseline]:
                status = "pair_plan_unavailable"
            elif len(wanted["model"]) != 1 or len(wanted[baseline]) != 1:
                status = "duplicate_plan"
            elif any(
                identities[id(r)] != identities[id(p)]
                for r, p in (
                    (left[0], wanted["model"][0]),
                    (right[0], wanted[baseline][0]),
                )
            ):
                status = "model_identity_mismatch"
            elif any(r.get("status") != "completed" for r in (left[0], right[0])):
                status = "not_completed"
            elif not all(_number(r.get("makespan")) for r in (left[0], right[0])):
                status = "missing_makespan"
            elif (
                not left[0].get("world_sha256")
                or len(
                    {
                        r.get("world_sha256")
                        for r in (
                            left[0],
                            right[0],
                            wanted["model"][0],
                            wanted[baseline][0],
                        )
                    }
                )
                != 1
            ):
                status = "world_mismatch"
            else:
                delta = left[0]["makespan"] - right[0]["makespan"]
            pairs.append(
                {
                    "case_id": case,
                    "replication": replication,
                    "baseline": baseline,
                    "training_seed": identities[id(left[0])][3]
                    if len(left) == 1
                    else None,
                    "checkpoint_sha256": identities[id(left[0])][2]
                    if len(left) == 1
                    else None,
                    "status": status,
                    "makespan_delta": delta,
                }
            )
    return {
        "id": directory.relative_to(root).as_posix(),
        "status": record.get("status", "unknown"),
        "stage": record.get("stage"),
        "error": record.get("error"),
        "requested": record.get("requested"),
        "coverage": coverage,
        "pairs": pairs,
        "input_coverage": record.get("input_coverage"),
        "warnings": sorted(warnings),
        "models": sorted(models),
        "runs": [
            {
                "run_dir": row["run_dir"],
                "case_id": row["case_id"],
                "algorithm_id": row["algorithm_id"],
                "replication": row["replication"],
                "training_seed": identities[id(row)][3],
                "checkpoint_sha256": identities[id(row)][2],
            }
            for row in observed
            if row.get("run_dir")
        ],
        "note": "Means use completed cases only. Replications of one saved model are not independent training seeds. Delta is model minus baseline on the same completed world.",
    }
