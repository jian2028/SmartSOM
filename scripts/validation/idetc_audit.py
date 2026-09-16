"""Grid study evidence audit; historical matrix replay stays source-bound."""

import json
import time
from pathlib import Path

from smartsom.trace.production import RUN_SCHEMA, audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_run(run_dir: Path, *, expected_jobs: int, expected_operations: int) -> dict:
    """Verify the recorded grid commands without creating replacement evidence."""
    started = time.monotonic()
    run_dir = Path(run_dir)
    path = run_dir / "run.json"
    require(
        path.is_file(),
        "historical matrix evidence requires its recorded source checkout for replay",
    )
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == RUN_SCHEMA, "expected grid run evidence")
    require(manifest["status"] == "completed", "run did not complete")
    scenario = manifest["inputs"]["scenario"]
    demands = scenario["demands"]
    require(
        len(demands) == expected_jobs
        and sum(len(d["steps"]) for d in demands) == expected_operations,
        "input coverage mismatch",
    )
    report = audit(run_dir)
    passed = manifest["result"]["completed"]
    require(
        set(passed) == {d["demand_id"] for d in demands},
        "unfinished qualified demand coverage",
    )
    rows = read_jsonl(run_dir / "trace.jsonl")
    holding = {
        b["buffer_id"] for b in scenario["factory"]["buffers"] if b["role"] == "storage"
    }
    holding_trips = sum(
        e["kind"] == "drop" and e.get("owner") in holding
        for r in rows
        for e in r["events"]
    )
    return {
        **report,
        "physical_model": "grid",
        "paper_comparison": "incompatible_physics",
        "makespan": manifest["result"]["tick"],
        "passed_job_ids": sorted(passed),
        "holding_trips": holding_trips,
        "checks": [
            "frozen_inputs",
            "state_hashes",
            "semantic_commands",
            "events",
            "qualified_demand_coverage",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
