"""Item-10 acceptance: configured evidence, exact replay and preregistered probabilities.

This bounded verification harness is not a batch experiment interface. Statistical
micro-episodes exercise Simulator directly and retain an aggregate, not 40,000
redundant run directories. All configured acceptance cases use run_one.
"""

import argparse
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

import yaml

from smartsom.algorithms import ScriptedPolicy
from smartsom.config import resolve_run
from smartsom.config.codec import digest
from smartsom.config.seeds import derive_seeds
from smartsom.dispatch import Dispatch
from smartsom.domain import Job, Operation, Order, ProcessingMode, WorkloadInstance
from smartsom.domain.quality import quality_mode_id
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.experiments import run_one
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.experiments.providers import build_provider
from smartsom.modules.quality import prepare_quality
from smartsom.workloads.quality import generate_quality

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference/quality/reference.json"


def check_run(resolved):
    attempt = run_one(resolved)
    result = attempt.simulation_result
    kwargs = dict(
        quality=resolved.quality,
        quality_probability_visibility=resolved.scenario.quality.probability_visibility,
        arrivals=resolved.arrivals,
        decision_trigger=resolved.scenario.decision_trigger,
        processing_times=resolved.processing_times,
        machine_events=resolved.machine_events,
        transport_enabled=resolved.transport_enabled,
        buffers_enabled=resolved.buffers_enabled,
    )
    assert (
        Simulator(resolved.factory, resolved.workload, **kwargs).run(
            build_provider(resolved.algorithm)
        )
        == result
    )
    assert (
        replay(resolved.factory, resolved.workload, result.actions, **kwargs) == result
    )
    schedule = (
        result.execution_schedule
        if resolved.transport_enabled or resolved.buffers_enabled
        else result.schedule
    )
    replayed = replay_schedule(resolved.factory, resolved.workload, schedule, **kwargs)
    assert replayed.execution_schedule == result.execution_schedule
    assert replayed.quality == result.quality and replayed.makespan == result.makespan

    def quality_events(trace):
        return tuple(
            replace(x, sequence=0)
            for x in trace
            if x.kind in ("quality_check", "inspection")
        )

    assert quality_events(result.trace) == quality_events(replayed.trace)
    return {
        "run_dir": str(attempt.run_dir),
        "makespan": result.makespan,
        "passing_rate": result.quality.passing_rate,
        "quality_sha256": digest(result.quality),
        "trace_sha256": digest(result.trace),
        "schedule_sha256": digest(result.execution_schedule),
    }


def configured(output):
    cases = {}
    for name in (
        "quality_m0",
        "quality_m1",
        "quality_m2",
        "quality_generated",
        "quality_hidden",
        "quality_machine",
        "quality_combined",
    ):
        resolved = resolve_run(ROOT / f"configs/runs/{name}.yaml")
        resolved = replace(
            resolved,
            run=resolved.run.model_copy(update={"output_root": str(output / "runs")}),
        )
        cases[name] = check_run(resolved)
    for name, expected in (("quality_m0", 24), ("quality_m1", 20), ("quality_m2", 16)):
        assert cases[name]["makespan"] == expected
    assert [
        cases[x]["passing_rate"] for x in ("quality_m0", "quality_m1", "quality_m2")
    ] == [1, 0, 0]
    base = yaml.safe_load(
        (ROOT / "configs/scenarios/quality_combined.yaml").read_text()
    )
    base["factory"] = str((ROOT / "configs/scenarios" / base["factory"]).resolve())
    for key in ("workload", "arrivals", "processing_time", "machine_events"):
        base[key]["path"] = str(
            (ROOT / "configs/scenarios" / base[key]["path"]).resolve()
        )
    for agv, buffers, ja, mb, upt in itertools.product((False, True), repeat=5):
        for trigger in (
            ("dispatch_available", "arrival_event") if ja else ("dispatch_available",)
        ):
            for policy in ("spt", "spt_quality_m0", "first_feasible"):
                name = f"AGV{int(agv)}-B{int(buffers)}-JA{int(ja)}-MB{int(mb)}-UPT{int(upt)}-{trigger}-{policy}"
                scenario = {
                    **base,
                    "transport": base["transport"] if agv else None,
                    "buffers": base["buffers"] if buffers else None,
                    "arrivals": base["arrivals"] if ja else None,
                    "machine_events": base["machine_events"] if mb else None,
                    "processing_time": base["processing_time"] if upt else None,
                    "decision_trigger": trigger,
                }
                path = output / f"{name}-scenario.yaml"
                path.write_text(yaml.safe_dump(scenario))
                run = output / f"{name}-run.yaml"
                run.write_text(
                    yaml.safe_dump(
                        {
                            "schema": "smartsom.run/v1",
                            "scenario": path.name,
                            "algorithm": str(
                                ROOT / f"configs/algorithms/{policy}.yaml"
                            ),
                            "seed": 42,
                            "output_root": str(output / "runs"),
                        }
                    )
                )
                cases[name] = check_run(resolve_run(run))
    # Exported base inputs and draws are independently reusable under another root.
    generated = Path(cases["quality_generated"]["run_dir"])
    scenario = {
        "schema": "smartsom.scenario/v1",
        "factory": str(ROOT / "configs/factories/quality.yaml"),
        "workload": {
            "kind": "instance",
            "path": str(generated / "realized_instance.json"),
        },
        "quality": {"kind": "fixed", "path": str(generated / "realized_quality.json")},
    }
    (output / "reimport-scenario.yaml").write_text(yaml.safe_dump(scenario))
    (output / "reimport-run.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "smartsom.run/v1",
                "scenario": "reimport-scenario.yaml",
                "algorithm": str(ROOT / "configs/algorithms/spt.yaml"),
                "seed": 999,
                "output_root": str(output / "runs"),
            }
        )
    )
    cases["reimport"] = check_run(resolve_run(output / "reimport-run.yaml"))
    for field in (
        "makespan",
        "passing_rate",
        "quality_sha256",
        "trace_sha256",
        "schedule_sha256",
    ):
        assert cases["reimport"][field] == cases["quality_generated"][field]
    return cases


def statistics(reference):
    spec = reference["statistics"]
    factory = resolve_run(ROOT / "configs/runs/quality_m0.yaml").factory
    workload = WorkloadInstance(
        (
            Order(
                "statistics",
                (
                    Job(
                        "route",
                        tuple(
                            Operation(
                                f"op_{i}",
                                (ProcessingMode("base", "M1", 10),),
                                (f"op_{i - 1}",) if i else (),
                            )
                            for i in range(5)
                        ),
                    ),
                ),
            ),
        )
    )
    rates = {
        x.quality_mode_id: float(x.error_rate)
        for x in factory.quality_speed.default_modes
    }
    counts = [0] * len(spec["routes"])
    actions = [
        tuple(
            Dispatch(f"op_{i}", quality_mode_id("base", mode))
            for i, mode in enumerate(route)
        )
        for route in spec["routes"]
    ]
    first = spec["root_seed_start"]
    size = spec["samples_per_group"]
    for index, root in enumerate(range(first, first + size)):
        seed = next(
            x.value
            for x in derive_seeds(root, generated=False, quality=True)
            if x.domain == "quality"
        )
        plan = prepare_quality(factory, workload, generate_quality(workload, seed))
        for group, script in enumerate(actions):
            result = Simulator(factory, workload, quality=plan).run(
                ScriptedPolicy(script)
            )
            counts[group] += result.quality.passed_jobs
        if (index + 1) % 1000 == 0:
            print(
                f"quality probability acceptance: {index + 1}/{size} seeds, four routes",
                flush=True,
            )
    report = []
    for route, count in zip(spec["routes"], counts, strict=True):
        expected = math.prod(1 - rates[mode] for mode in route)
        observed = count / size
        report.append(
            {
                "route": route,
                "passed": count,
                "samples": size,
                "expected": expected,
                "observed": observed,
                "absolute_error": abs(observed - expected),
                "tolerance": spec["absolute_tolerance"],
            }
        )
    assert all(row["absolute_error"] <= row["tolerance"] for row in report), report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Configured smoke only; does not satisfy item-10 probability acceptance.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(REFERENCE.read_text())
    report = {
        "source": source_identity(),
        "reference_sha256": digest(reference),
        "statistics_spec": reference["statistics"],
    }
    report["configured"] = configured(output)
    write_json(output / "acceptance.json", report)
    print(f"configured + replay passed: {len(report['configured'])} runs", flush=True)
    report["statistics"] = None if args.skip_statistics else statistics(reference)
    report["complete"] = not args.skip_statistics
    write_json(output / "acceptance.json", report)
    print(
        json.dumps(
            {
                "report": str(output / "acceptance.json"),
                "complete": report["complete"],
                "statistics": report["statistics"],
            }
        )
    )


if __name__ == "__main__":
    main()
