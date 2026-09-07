"""Bounded item-8 acceptance: configured runs and both exact replay routes.

No optional dependency is needed. Each configuration variant is materialized
through resolve_run and executed through run_one, with source-identified evidence.
Use --output-dir to choose a new local directory (existing directories rejected).
"""

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import yaml

from smartsom.config import resolve_run
from smartsom.config.codec import digest, primitive
from smartsom.config.models import ExecutionScheduleFile
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.experiments import run_one
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.providers import build_provider

ROOT = Path(__file__).resolve().parents[1]


def check_run(resolved, *, expected=None):
    result = run_one(resolved)
    kwargs = dict(
        arrivals=resolved.arrivals,
        decision_trigger=resolved.scenario.decision_trigger,
        processing_times=resolved.processing_times,
        machine_events=resolved.machine_events,
        transport_enabled=True,
    )
    simulation = result.simulation_result
    direct = Simulator(resolved.factory, resolved.workload, **kwargs).run(
        build_provider(resolved.algorithm)
    )
    assert direct == simulation
    assert (
        replay(resolved.factory, resolved.workload, simulation.actions, **kwargs)
        == simulation
    )
    stored = ExecutionScheduleFile.model_validate_json(
        (result.run_dir / "execution_schedule.json").read_text()
    ).execution_schedule
    replayed = replay_schedule(resolved.factory, resolved.workload, stored, **kwargs)
    assert (
        replayed.execution_schedule == simulation.execution_schedule
        and replayed.makespan == simulation.makespan
    )
    assert (
        replay_schedule(
            resolved.factory,
            resolved.workload,
            replace(
                stored,
                transports=tuple(reversed(stored.transports)),
                operations=tuple(reversed(stored.operations)),
            ),
            **kwargs,
        )
        == replayed
    )
    if expected is not None:
        assert simulation.makespan == expected
    return {
        "run_dir": str(result.run_dir),
        "makespan": simulation.makespan,
        "schedule_sha256": digest(simulation.execution_schedule),
        "trace_sha256": digest(simulation.trace),
        "delivered_jobs": sum(
            t.destination.kind == "output" for t in simulation.transport_schedule
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"source": source_identity(), "cases": {}}
    for case, expected in (("transport_hand", 15), ("transport_reroute", 7)):
        resolved = resolve_run(ROOT / f"configs/runs/{case}.yaml")
        resolved = replace(
            resolved,
            run=resolved.run.model_copy(update={"output_root": str(output / "runs")}),
        )
        report["cases"][case] = check_run(resolved, expected=expected)
        if case == "transport_hand":
            reference = ExecutionScheduleFile.model_validate_json(
                (ROOT / "data/reference/transport/hand_schedule.json").read_text()
            ).execution_schedule
            actual = replay_schedule(
                resolved.factory, resolved.workload, reference, transport_enabled=True
            )
            assert actual.makespan == 15 and actual.execution_schedule == reference
    base = yaml.safe_load(
        (ROOT / "configs/scenarios/transport_combined.yaml").read_text()
    )
    # Convert source references before placing variants beside their own run files.
    base["factory"] = str((ROOT / "configs/scenarios" / base["factory"]).resolve())
    for key in ("workload", "arrivals", "processing_time", "machine_events"):
        base[key]["path"] = str(
            (ROOT / "configs/scenarios" / base[key]["path"]).resolve()
        )
    for ja, mb, upt in itertools.product((False, True), repeat=3):
        for trigger in (
            ("dispatch_available", "arrival_event") if ja else ("dispatch_available",)
        ):
            for policy in ("spt_transport", "first_feasible"):
                name = f"JA{int(ja)}-MB{int(mb)}-UPT{int(upt)}-{trigger}-{policy}"
                scenario = {
                    **base,
                    "arrivals": base["arrivals"] if ja else None,
                    "machine_events": base["machine_events"] if mb else None,
                    "processing_time": base["processing_time"] if upt else None,
                    "decision_trigger": trigger,
                }
                scenario_path = output / f"{name}-scenario.yaml"
                scenario_path.write_text(yaml.safe_dump(scenario))
                run_path = output / f"{name}-run.yaml"
                run_path.write_text(
                    yaml.safe_dump(
                        {
                            "schema": "smartsom.run/v1",
                            "scenario": scenario_path.name,
                            "algorithm": str(
                                ROOT / f"configs/algorithms/{policy}.yaml"
                            ),
                            "seed": 42,
                            "output_root": str(output / "runs"),
                        }
                    )
                )
                report["cases"][name] = check_run(resolve_run(run_path))
    write_json(output / "acceptance.json", report)
    print(
        json.dumps(
            {
                "report": str(output / "acceptance.json"),
                "runs": len(report["cases"]),
                "source": primitive(report["source"]["git"]),
            }
        )
    )


if __name__ == "__main__":
    main()
