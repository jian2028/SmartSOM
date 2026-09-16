"""Audit evaluation coverage and child grid trajectories without rerunning policies."""

import json
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments.evaluation import _summary
from smartsom.experiments.packaging import locate_reference


def audit_tree(source, seen=None):
    from smartsom.trace.production import RUN_SCHEMA, audit

    directory = Path(source).expanduser().resolve()
    if directory.is_file():
        directory = directory.parent
    seen = set() if seen is None else seen
    if directory in seen:
        raise ValueError("cyclic evaluation audit reference")
    seen.add(directory)
    owner = directory / "run.json"
    record = json.loads(owner.read_text())
    if record.get("schema") == RUN_SCHEMA:
        return audit(directory)
    if record.get("schema") == "smartsom.experiment/v2":
        target = record.get("paths", {}).get("evaluation")
        if not target:
            raise ValueError("experiment has no evaluation evidence to audit")
        return {
            **audit_tree(locate_reference(owner, target), seen),
            "experiment": str(directory),
        }
    if record.get("schema") != "smartsom.evaluation/v1":
        raise ValueError("historical artifacts are not grid execution evidence")
    if record.get("status") not in ("completed", "completed_with_failures", "failed"):
        raise ValueError("evaluation is unfinished")
    plan_path = locate_reference(
        owner, record.get("paths", {}).get("plan", "plan.json")
    )
    planned = json.loads(plan_path.read_text())["entries"]
    rows = record.get("results", [])
    if not rows or len(rows) != len(planned) or len(rows) != record.get("requested"):
        raise ValueError("evaluation plan/result coverage mismatch")
    keys = [(r["case_id"], r["replication"], r["algorithm_id"]) for r in planned]
    cases = {r["case_id"] for r in record["cases"]}
    algorithms = {r["algorithm_id"] for r in planned}
    expected = {
        (case, replication, algorithm)
        for case in cases
        for replication in range(record["options"]["replications"])
        for algorithm in algorithms
    }
    if (
        len(algorithms) != 1 + len(record["options"]["baselines"])
        or set(keys) != expected
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("evaluation replication/algorithm coverage mismatch")
    worlds, children, results = {}, set(), []
    for entry, row in zip(planned, rows, strict=True):
        if any(row.get(key) != value for key, value in entry.items()):
            raise ValueError("evaluation planned identity mismatch")
        pair = row["case_id"], row["replication"]
        world = row["world_seed"], row["world_sha256"]
        if worlds.setdefault(pair, world) != world:
            raise ValueError("evaluation paired world mismatch")
        if row.get("run_dir") is None:
            results.append(
                {
                    "status": "unavailable",
                    "error": "no child run evidence",
                    "algorithm_id": row["algorithm_id"],
                }
            )
            continue
        child = locate_reference(owner, row["run_dir"])
        if child in children:
            raise ValueError("duplicate evaluation child directory")
        children.add(child)
        manifest = json.loads((child / "run.json").read_text())
        if manifest.get("schema") != RUN_SCHEMA:
            raise ValueError("evaluation child uses a different execution contract")
        if (
            digest(manifest["inputs"]["scenario"]) != row["world_sha256"]
            or manifest["seed"] != row["world_seed"]
            or manifest["provider"] != row["provider"]
        ):
            raise ValueError("evaluation run input/provider mismatch")
        if manifest.get("checkpoint") != row.get("checkpoint"):
            raise ValueError("evaluation checkpoint identity mismatch")
        if manifest.get("algorithm_seed") != row.get("algorithm_seed"):
            raise ValueError("evaluation algorithm seed mismatch")
        if manifest["status"] != row["status"]:
            raise ValueError("evaluation child status mismatch")
        state = manifest["result"]
        makespan = state["tick"] if row["status"] == "completed" else None
        if (
            row.get("makespan") != makespan
            or ("return" in row and row["return"] != state["return"])
            or (
                "passing_rate" in row
                and row["passing_rate"]
                != len(state["completed"])
                / max(1, len(manifest["inputs"]["scenario"]["demands"]))
            )
        ):
            raise ValueError("evaluation result differs from child evidence")
        if manifest.get("trace") is None:
            # An in-memory audit was possible during execution; it is not
            # reproducible execution evidence after the process has exited.
            result = {"status": "unavailable", "error": "trajectory was not recorded"}
        else:
            result = audit(child)
        results.append({"run_dir": row["run_dir"], **result})
    summary = _summary(rows, len(planned))
    if any(record.get(key) != value for key, value in summary.items()):
        raise ValueError("evaluation summary counts mismatch")
    status = (
        "failed"
        if summary["engineering_failures"]
        else "completed_with_failures"
        if summary["failed"]
        else "completed"
    )
    if record["status"] != status:
        raise ValueError("evaluation final status mismatch")
    saved = json.loads(locate_reference(owner, record["paths"]["summary"]).read_text())
    if saved.get("status") != status or any(
        saved.get(key) != value for key, value in summary.items()
    ):
        raise ValueError("evaluation summary artifact mismatch")
    return {
        "status": "passed"
        if all(r["status"] == "passed" for r in results)
        else "partial_verified",
        "evaluation": str(directory),
        "checks": len(rows),
        "runs": results,
    }
