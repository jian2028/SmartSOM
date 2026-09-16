"""Historical interval references; this script does not execute the grid simulator.

Run from the checkout with uv run --no-sync python scripts/validate_machine_events.py.
Add --cp for pinned PyJobShop and --dsbx-root PATH for the pinned external source.
The latter uses the separate validation requirements, not the project dependencies.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from smartsom.algorithms.reference_schedule import validate_schedule
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    MachineOutage,
    MachineOutagePlan,
    Operation,
    Order,
    ProcessingMode,
    ScheduledOperation,
    WorkloadInstance,
)
from smartsom.experiments.evidence import source_identity, write_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/reference/machine_events/cases.json"


@dataclass(frozen=True)
class IntervalReference:
    makespan: int
    schedule: tuple
    kind: str = "historical_interval_validation"


def interval_reference(case):
    factory = FactorySpec((Machine("M1"),))
    workload = WorkloadInstance(
        (
            Order(
                "order",
                tuple(
                    Job(
                        f"J{i}",
                        (
                            Operation(
                                f"O{i}", (ProcessingMode("standard", "M1", ticks),)
                            ),
                        ),
                    )
                    for i, ticks in enumerate(case["durations"])
                ),
            ),
        )
    )
    plan = MachineOutagePlan(
        tuple(MachineOutage("M1", *window) for window in case["outages"])
    )
    schedule = tuple(
        ScheduledOperation(f"O{i}", "standard", "M1", start, end)
        for i, (start, end) in enumerate(zip(case["starts"], case["ends"]))
    )
    validated = validate_schedule(factory, workload, schedule, machine_events=plan)
    segments = []
    for entry in schedule:
        active = [
            tick
            for tick in range(entry.start_time, entry.completion_time)
            if not any(start <= tick < end for start, end in case["outages"])
        ]
        spans = []
        for tick in active:
            if spans and spans[-1][1] == tick:
                spans[-1][1] = tick + 1
            else:
                spans.append([tick, tick + 1])
        segments.append(spans)
    assert segments == case["segments"]
    makespan = max(row.completion_time for row in validated)
    assert makespan == case["makespan"]
    return IntervalReference(makespan, validated)


def pyjobshop_reference(case):
    from pyjobshop import Model

    assert importlib.metadata.version("pyjobshop") == "0.0.9"
    assert importlib.metadata.version("ortools") == "9.12.4544"
    model = Model()
    machine = model.add_machine(breaks=[tuple(window) for window in case["outages"]])
    for i, ticks in enumerate(case["durations"]):
        task = model.add_task(
            earliest_start=case["starts"][i],
            latest_start=case["starts"][i],
            allow_breaks=True,
        )
        model.add_mode(task, machine, ticks)
    solution = model.solve(
        solver="ortools", time_limit=10, num_workers=1, random_seed=0, display=False
    )
    assert solution.status.name == "OPTIMAL"
    rows = [
        {"start": task.start, "end": task.end, "breaks": task.breaks}
        for task in solution.best.tasks
    ]
    assert [row["start"] for row in rows] == case["starts"]
    assert [row["end"] for row in rows] == case["ends"]
    assert solution.objective == solution.lower_bound == case["makespan"]
    # Independently intersect the returned task spans with machine uptime.
    segments = []
    for row in rows:
        cursor, parts = row["start"], []
        for start, end in case["outages"]:
            if start >= row["end"]:
                break
            if end <= cursor:
                continue
            if cursor < start:
                parts.append([cursor, start])
            cursor = end
        if cursor < row["end"]:
            parts.append([cursor, row["end"]])
        segments.append(parts)
    assert segments == case["segments"]
    return {
        "status": str(solution.status),
        "objective": solution.objective,
        "bound": solution.lower_bound,
        "tasks": rows,
        "segments": segments,
    }


def dsbx_references(root, cases, expected_commit):
    requirements = ROOT / "scripts/validation/dynaschedbench-requirements.txt"
    for line in requirements.read_text().splitlines():
        if line and not line.startswith("#"):
            name, expected = line.split("==")
            assert importlib.metadata.version(name) == expected, name
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == expected_commit
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    sys.path.insert(0, str(root / "src"))
    from dsbx.Gen import InputModel
    from dsbx.Gen.models.events import (
        ArrivalEvent,
        BreakdownEvent,
        MachineRepairCompletionEvent,
    )
    from dsbx.Sim import DynaSchedSim
    from loguru import logger

    logger.remove()
    model = InputModel.model_validate(
        {
            "plant": {
                "machines": [{"id": "M1", "group": "G1", "speed": 1}],
                "process_templates": [
                    {
                        "family": "F",
                        "route": [
                            {
                                "machine_group": "G1",
                                "process_time": {"dist": "const", "mean": 5},
                            }
                        ],
                    }
                ],
            },
            "scale": {"horizon": 20, "jobs_total": 1},
        }
    )
    outputs = []
    for case in cases:
        if len(case["durations"]) != 1:
            continue
        events = [
            ArrivalEvent(
                time=0, job_id="A", job_family="F", routing=["G1"], process_times=[5]
            )
        ]
        for start, end in case["outages"]:
            events.extend(
                [
                    BreakdownEvent(time=start, machine_id="M1", duration=end - start),
                    MachineRepairCompletionEvent(time=end, machine_id="M1"),
                ]
            )
        simulator = DynaSchedSim(model, events)
        simulator.reset()
        dispatches = []
        for _ in range(12):
            if simulator.get_ready_operations():
                dispatches.append(simulator.state.time)
                simulator.step_action(
                    {"job_id": "A", "machine_group": "G1", "machine_id": "M1"}
                )
            before = simulator.state.time
            simulator.advance_to_next_decision_point()
            if before == simulator.state.time:
                break
        parts = [
            [row.start, row.end]
            for row in simulator.state.machines["M1"].schedule_segments
            if row.job_id == "A"
        ]
        completion = simulator.state.jobs["A"].completion_time
        assert parts == case["segments"][0] and completion == case["makespan"]
        assert dispatches == [start for start, _ in parts]
        outputs.append(
            {
                "id": case["id"],
                "dispatch_ticks": dispatches,
                "processing_segments": parts,
                "completion": completion,
                "raw_get_gantt": simulator.get_gantt(),
            }
        )
    frozen = json.loads((FIXTURE.parent / "dynaschedbench.json").read_text())
    assert outputs == frozen["cases"]
    return {
        "source_commit": commit,
        "source_file_sha256": hashlib.sha256(
            (root / "src/dsbx/Sim/Simulator.py").read_bytes()
        ).hexdigest(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "pydantic",
                "numpy",
                "pandas",
                "loguru",
                "seaborn",
                "scipy",
                "jinja2",
                "matplotlib",
            )
        },
        "cases": outputs,
        "limitation": "Controller explicitly redispatches after repair. Raw get_gantt retains pre-interruption projected ends; compare actual machine schedule_segments, not action equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--dsbx-root", type=Path)
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text())
    report = {
        "source": source_identity(),
        "reference_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in fixture["cases"]:
        result = interval_reference(case)
        row = {"id": case["id"], "result": result}
        if args.cp:
            row["pyjobshop"] = pyjobshop_reference(case)
        report["cases"].append(row)
    if args.dsbx_root:
        report["dynaschedbench"] = dsbx_references(
            args.dsbx_root.resolve(), fixture["cases"], fixture["dynaschedbench_commit"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "makespans": [row["result"].makespan for row in report["cases"]],
                "cp": args.cp,
                "dynaschedbench": bool(args.dsbx_root),
            }
        )
    )


if __name__ == "__main__":
    main()
