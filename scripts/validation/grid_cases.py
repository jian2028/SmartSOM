"""Development checks for configured grid cases, not historical acceptance scores."""

import argparse
import json
from pathlib import Path

from smartsom.api import load_config, prepare
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.runner import run_one
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[2]


def check_run(prepared, output):
    attempt = run_one(prepared, output_root=output, verbose=False)
    checked = audit(attempt.run_dir)
    result = attempt.simulation_result
    expected = {d.demand_id for d in prepared.resolved.scenario.demands}
    completed = set(result.final_state["completed"])
    return {
        "run_dir": str(attempt.run_dir),
        "makespan": result.makespan,
        "status": result.status,
        "qualified_demands": len(completed),
        "required_demands": len(expected),
        "execution_replay": checked,
        "passed": result.status == "completed"
        and completed == expected
        and checked["status"] == "passed",
    }


def main(cases, *, description, statistics=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = {
        name: prepare(load_config(ROOT / f"configs/runs/{name}.yaml"), training=False)
        for name in cases
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "smartsom.grid-development-checks/v1",
        "evidence_kind": "development_verification",
        "source": source_identity(),
        "cases": {},
    }
    for name, case in prepared.items():
        try:
            report["cases"][name] = check_run(case, output / "runs")
        except Exception as exc:
            report["cases"][name] = {"passed": False, "error": str(exc)}
        write_json(output / "acceptance.json", report)
    if statistics:
        report["draw_distribution"] = statistics()
    report["passed"] = all(row["passed"] for row in report["cases"].values())
    if statistics:
        report["passed"] &= all(row["passed"] for row in report["draw_distribution"])
    write_json(output / "acceptance.json", report)
    print(
        json.dumps(
            {"report": str(output / "acceptance.json"), "passed": report["passed"]}
        )
    )
    return 0 if report["passed"] else 1
