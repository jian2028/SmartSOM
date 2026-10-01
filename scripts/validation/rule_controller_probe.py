"""Local engineering probes for rule throughput; these are not formal experiments."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

from smartsom.algorithms.agv_planner import action_destination, plan_actions
from smartsom.algorithms.production_composition import BoundaryCoordinator
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.config.production import ScenarioFile, WorkloadFile, materialize
from smartsom.engine.production import ProductionSimulator
from smartsom.studio.templates import load_template_file


def probe(
    number, jobs, seed, active, limit, admission="sqrt", *, record_directory=None
):
    source_identity = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "engineering_probe": True,
        "working_tree": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines(),
        "code_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/smartsom/algorithms/production_rules.py",
                "src/smartsom/algorithms/agv_planner.py",
                "src/smartsom/algorithms/agv_dispatcher.py",
                "src/smartsom/algorithms/rule_registry.py",
                "scripts/validation/rule_controller_probe.py",
            )
        },
    }
    factory = load_template_file(number).factory
    workload = WorkloadFile.model_validate(
        {
            "schema": "smartsom.workload/v2",
            "profile": {
                "jobs": jobs,
                "route": tuple(f"operation_{i}" for i in range(1, 5)),
                "nominal_min": 1,
                "nominal_max": 1,
                "due_at": 1000,
                "input_id": "buffer_001",
            },
        }
    )
    settings = ScenarioFile.model_validate(
        {
            "schema": "smartsom.scenario/v2",
            "factory": "unused",
            "workload": "unused",
            "mode": "finite",
            "tick_limit": limit,
        }
    )
    sim = ProductionSimulator(
        materialize(factory, workload, settings, seed), contract="v3"
    )
    parameters = {"fleet_admission": admission, "work_in_progress_first": True}
    if active:
        parameters["max_active"] = active
    policies = {
        "machine": RulePolicy("machine", "normal_first", seed=11),
        "buffer": RulePolicy("buffer", "edd", seed=12),
        "dispatcher": RulePolicy(
            "dispatcher", "nearest", seed=13, parameters=parameters
        ),
        "mover": RulePolicy("mover", "clearance_shortest_path", seed=14),
    }
    coordinator = BoundaryCoordinator(
        sim,
        policies,
        {role: {"default": role, "overrides": {}} for role in policies},
        matching="priority_greedy",
    )
    recorder = None
    if record_directory is not None:
        from smartsom.algorithms.rule_registry import freeze_rule
        from smartsom.config.codec import primitive
        from smartsom.domain.production_decisions import ACTION_CONTRACT
        from smartsom.trace.production import Recorder, atomic_json

        recorder = Recorder(
            record_directory,
            {
                "scenario": primitive(sim.scenario),
                "algorithm": {"provider": "composable"},
                "composition": {
                    "matching": "priority_greedy",
                    "policies": {
                        role: {
                            "seed": policy.seed,
                            "identity": freeze_rule(
                                role, policy.name, parameters=policy.parameters
                            ),
                        }
                        for role, policy in policies.items()
                    },
                },
            },
            sim.snapshot(),
            source=source_identity,
        )
        recorder.manifest.update(
            action_contract=ACTION_CONTRACT,
            context={
                "case": f"Optimized Template {number}",
                "seed": seed,
                "replication": 1,
            },
        )
        atomic_json(recorder.directory / "run.json", recorder.manifest)
    rejections, previous, reversals = Counter(), {}, Counter()
    recent_moves, longer_cycles = {}, 0
    moves, cycles, maximum_active = 0, 0, 0
    start = perf_counter()
    while not sim.done:
        row = coordinator.tick()
        if recorder is not None:
            recorder.append(row)
        rejections.update(row["rejections"].values())
        maximum_active = max(
            maximum_active,
            sum(
                bool(a["target"] or a["job"] or a["reservation"])
                for a in sim.agvs.values()
            ),
        )
        for event in row["events"]:
            if event["kind"] != "move":
                continue
            moves += 1
            owner = event["agv"]
            state = row["boundary_state"]["agvs"][owner]
            task = (state["target"], state.get("job"), state.get("reservation"))
            prior = previous.get(owner)
            if prior and prior == (task, event["after"], event["before"]):
                reversals[owner] += 1
                cycles += reversals[owner] >= 2
            else:
                reversals[owner] = 0
            previous[owner] = (task, event["before"], event["after"])
            recent_task, cells = recent_moves.get(owner, (None, []))
            if recent_task != task:
                cells = []
            cells = [*cells, tuple(event["after"])][-16:]
            longer_cycles += len(cells) == 16 and len(set(cells)) <= 4
            recent_moves[owner] = task, cells
        if sim.tick % 200 == 0:
            print(
                f"template={number} seed={seed} active={active} tick={sim.tick} completed={len(sim.completed)}",
                flush=True,
            )
    if recorder is not None:
        recorder.finish(sim.status)
    return {
        "source": source_identity,
        "template": number,
        "jobs": jobs,
        "seed": seed,
        "max_active": active,
        "admission": admission,
        "recording": str(recorder.directory.resolve())
        if recorder is not None
        else None,
        "status": sim.status,
        "tick": sim.tick,
        "completed": len(sim.completed),
        "seconds": round(perf_counter() - start, 3),
        "moves": moves,
        "persistent_reversals": cycles,
        "longer_cycles": longer_cycles,
        "peak_active": maximum_active,
        "rejections": dict(rejections),
        "metrics": dict(sim.metrics),
        "final_agvs": sim.snapshot(public=True)["agvs"]
        if sim.status != "completed"
        else {},
        "final_jobs": sim.snapshot(public=True)["jobs"]
        if sim.status != "completed"
        else {},
        "final_actions": policies["mover"]._mover_plan
        if sim.status != "completed"
        else {},
        "final_history": policies["mover"]._mover_history
        if sim.status != "completed"
        else {},
        "final_view": sim.protocol.public_view() if sim.status != "completed" else {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[202])
    parser.add_argument("--active", nargs="+", type=int, default=[1])
    parser.add_argument("--admission", default="sqrt")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", type=Path)
    parser.add_argument(
        "--record-root",
        type=Path,
        help="Save full visual replay traces in new subdirectories",
    )
    args = parser.parse_args()
    if args.inspect:
        report = json.loads(args.inspect.read_text(encoding="utf-8"))[0]
        factory = load_template_file(report["template"]).factory
        # Geometry is static public input; no physics is executed in this mode.
        workload = WorkloadFile.model_validate(
            {
                "schema": "smartsom.workload/v2",
                "profile": {
                    "jobs": 1,
                    "route": ("operation_1",),
                    "nominal_min": 1,
                    "nominal_max": 1,
                    "due_at": 1000,
                    "input_id": "buffer_001",
                },
            }
        )
        settings = ScenarioFile.model_validate(
            {
                "schema": "smartsom.scenario/v2",
                "factory": "unused",
                "workload": "unused",
                "mode": "finite",
                "tick_limit": 2000,
            }
        )
        sim = ProductionSimulator(
            materialize(factory, workload, settings, 202), contract="v3"
        )
        observation = sim.protocol.public_view()
        observation.update(agvs=report["final_agvs"], tick=report["tick"], sources={})
        history = report.get("final_history", {})
        plan = plan_actions(observation, history)
        occupied = {tuple(a["cell"]): owner for owner, a in observation["agvs"].items()}
        print(json.dumps({"plan": plan}, indent=2))
        for owner, action in plan.items():
            if action != "WAIT":
                cell = action_destination(observation["agvs"], owner, action)
                print(
                    owner,
                    observation["agvs"][owner]["cell"],
                    action,
                    cell,
                    "occupied",
                    occupied.get(cell),
                )
        return
    if args.output is None:
        parser.error("--output is required for a probe")
    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for number in args.templates:
        for seed in args.seeds:
            for active in args.active:
                recording = (
                    args.record_root
                    / f"template_{number:03d}_seed_{seed}_active_{active}"
                    if args.record_root is not None
                    else None
                )
                result = probe(
                    number,
                    args.jobs,
                    seed,
                    active,
                    args.limit,
                    args.admission,
                    record_directory=recording,
                )
                reports.append(result)
                print(
                    json.dumps(
                        {k: v for k, v in result.items() if not k.startswith("final_")}
                    ),
                    flush=True,
                )
                args.output.write_text(
                    json.dumps(reports, indent=2) + "\n", encoding="utf-8"
                )


if __name__ == "__main__":
    main()
