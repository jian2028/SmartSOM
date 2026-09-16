"""Repeat local gates before full static and isolated/combined disturbances."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from template1_audit import check
from template1_evidence import capture_source

from smartsom import api
from smartsom.domain.production import Outage
from smartsom.experiments.production import execute
from smartsom.trace.production import atomic_json, audit

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = (
        ROOT
        / "artifacts/template1"
        / ("validation-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    capture_source(output / "source")
    warmup, static, disturbed = [
        api.prepare(
            api.load_config(ROOT / f"configs/runs/template1_{name}.yaml"),
            training=False,
        ).resolved
        for name in ("warmup", "static", "disturbed")
    ]
    cases = [
        ("warmup", warmup.scenario),
        (
            "local_pause",
            replace(warmup.scenario, outages=(Outage("machine_001", 10, 13),)),
        ),
        ("static", static.scenario),
        ("arrival_only", replace(disturbed.scenario, outages=())),
        (
            "outage_only",
            replace(disturbed.scenario, demands=disturbed.scenario.demands[:12]),
        ),
        ("disturbed", disturbed.scenario),
    ]
    report = {"kind": "development_verification", "cases": {}}
    for name, scenario in cases:
        directory = execute(
            scenario, static.algorithm, output_root=output, name=name, verbose=False
        )
        ledger = check(directory)
        replay = audit(directory)
        assert replay["status"] == "passed"
        if name == "local_pause":
            first = ledger["samples"][0]
            assert (first["start"], first["end"], first["active_ticks"]) == (9, 16, 4)
        report["cases"][name] = {
            "run_dir": str(directory),
            "ledger": ledger,
            "replay": replay,
        }
        atomic_json(output / "report.json", report)
        print(name, ledger["qualified"], flush=True)
    report["passed"] = True
    atomic_json(output / "report.json", report)
    for name in ("static", "disturbed"):
        link = ROOT / f"artifacts/template1/{name}-replay"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(report["cases"][name]["run_dir"], target_is_directory=True)
    atomic_json(
        ROOT / "artifacts/template1/latest-validation.json",
        {"report": str(output / "report.json")},
    )
    print(json.dumps({"report": str(output / "report.json"), "passed": True}))


if __name__ == "__main__":
    main()
