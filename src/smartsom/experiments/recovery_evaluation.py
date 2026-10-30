"""Explicit candidate calibration and frozen controlled-fault evaluation."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean

from smartsom.algorithms.base_policy import (
    build_policy,
    evaluation_recipe,
    uses_learning,
)
from smartsom.config.base_diagnostic import generate_scenario, load_inputs, seed_for
from smartsom.config.codec import primitive
from smartsom.domain.production import Outage
from smartsom.experiments.evaluation_resume import CaseStore
from smartsom.telemetry.recovery_protocol import platform, recovery_evidence
from smartsom.telemetry.research_metrics import fingerprint


def run(args):
    from smartsom.experiments import reservation_runtime as rt
    from smartsom.telemetry.experiment_tracking import ExperimentTracking

    ex, factory, workload, algorithm, paths = load_inputs(args.experiment)
    settings = ex.evaluation.recovery
    root = Path(args.output)
    methods = args.method or ["rule"]
    cases = args.cases or (
        ex.validation_cases if args.split == "validation" else ex.test_cases
    )
    if ex.run.task != "calibrate" and cases % 4:
        raise ValueError(
            "recovery world inventory derives from balanced four-condition case counts"
        )
    world_count = cases if ex.run.task == "calibrate" else cases // 4
    repeat_count = args.action_seeds
    candidates = (
        settings.candidates
        if settings.action == "calibrate"
        else tuple(c for c in settings.candidates if c.name == settings.selected)
    )
    learning_methods = any(
        uses_learning(evaluation_recipe(algorithm, method)) for method in methods
    )
    checkpoint = rt.checkpoint_training_evidence(args) if learning_methods else None
    if learning_methods and checkpoint is None:
        raise ValueError(
            "recovery learning policies require an explicit frozen checkpoint"
        )
    definition = {
        "protocol": settings.model_dump(
            mode="json",
            exclude={"action", "selected", "calibration", "calibration_sha256"},
        ),
        "inputs": {
            k: hashlib.sha256(paths[k].read_bytes()).hexdigest()
            for k in ("factory", "workload", "algorithm")
        },
        "selection": args.selection,
        "methods": methods,
        "checkpoint_sha256": checkpoint,
    }
    reference = None
    if settings.action == "evaluate":
        reference_path = (paths["experiment"].parent / settings.calibration).resolve()
        raw_reference = reference_path.read_bytes()
        if (
            settings.calibration_sha256
            and hashlib.sha256(raw_reference).hexdigest() != settings.calibration_sha256
        ):
            raise ValueError("frozen recovery calibration changed")
        reference = json.loads(raw_reference)
        if reference.get("definition_sha256") != fingerprint(definition):
            raise ValueError(
                "recovery calibration definition differs from frozen inputs"
            )
        if not reference.get("accepted", {}).get(settings.selected, False):
            raise ValueError(
                "selected recovery candidate has no valid frozen calibration"
            )
    source = rt.source_identity()
    if (
        reference is not None
        and reference.get("source", {}).get("implementation_sha256")
        != source["implementation_sha256"]
    ):
        raise ValueError("recovery calibration source differs; use its original source")
    manifest = rt.freeze(
        root,
        paths,
        "evaluate",
        {
            "recovery_definition": definition,
            "reference_sha256": fingerprint(reference) if reference else None,
            "checkpoint_sha256": checkpoint,
            "training_progress": getattr(args, "training_progress", None),
            "methods": methods,
            "tracking_group": getattr(args, "tracking_group", None),
        },
    )
    tracking = getattr(args, "tracking", None) or ExperimentTracking(
        root,
        ex.tracking,
        job_type="calibration" if settings.action == "calibrate" else "evaluate",
    )
    args._tracking = tracking
    if not getattr(args, "tracking", None):
        tracking.start(manifest, ex.model_dump(mode="json", by_alias=True))
    entries, worlds = [], []
    for candidate in candidates:
        work = workload.model_copy(
            update={
                "orders": candidate.orders,
                "initial_orders": candidate.initial_orders,
                "mean_interarrival": candidate.mean_interarrival,
            }
        )
        for method in methods:
            for world_index in range(world_count):
                seed = seed_for(
                    90000 if args.split == "test" else 80000, world_index
                ) % (2**31)
                nominal = replace(
                    generate_scenario(factory, work, seed, "normal", ex.drain_ticks),
                    tick_limit=candidate.horizon_ticks,
                )
                for repeat in range(repeat_count):
                    for fault_index, fault in enumerate(candidate.faults):
                        for disturbed in (
                            (False,)
                            if settings.action == "calibrate"
                            else (False, True)
                        ):
                            scenario = (
                                replace(
                                    nominal,
                                    outages=(
                                        Outage(
                                            fault.machine_id,
                                            fault.start_tick,
                                            fault.start_tick + fault.duration_ticks,
                                        ),
                                    ),
                                )
                                if disturbed
                                else nominal
                            )
                            entry = {
                                "case": f"recovery-{len(entries)}",
                                "replication": 0,
                                "action_seed": repeat,
                                "seed": seed,
                                "scenario": primitive(scenario),
                                "candidate": candidate.name,
                                "method": method,
                                "world_index": world_index,
                                "fault_index": fault_index,
                                "disturbed": disturbed,
                            }
                            entries.append(entry)
                            worlds.append((scenario, work, candidate, fault))
    store = CaseStore(
        root / "case-commits",
        None,
        entries,
        "recovery",
        context={
            "definition": definition,
            "source": source["implementation_sha256"],
            "checkpoint": checkpoint,
            "methods": methods,
            "reference": fingerprint(reference) if reference else None,
        },
    )
    deadline = time.monotonic() + max(
        0, manifest["started_unix"] + args.seconds - time.time()
    )
    if getattr(args, "deadline_unix", None) is not None:
        deadline = min(
            deadline, time.monotonic() + max(0, args.deadline_unix - time.time())
        )
    rows = []
    for index, entry in enumerate(entries):
        if index in store.rows:
            rows.append(store.rows[index]["payload"])
            continue
        scenario, work, candidate, fault = worlds[index]
        options = evaluation_recipe(algorithm, entry["method"])
        learning = uses_learning(options)
        policy_seed = seed_for(entry["seed"], entry["action_seed"]) % (2**31)
        policy = build_policy(
            options,
            policy_seed,
            training=False,
            selection=args.selection,
            numerical_threads=getattr(args, "numerical_threads", 1),
        )
        if learning:
            if (
                hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
                != checkpoint
            ):
                raise ValueError("checkpoint changed during recovery evaluation")
            policy.load(args.checkpoint)
            if (
                hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
                != checkpoint
            ):
                raise ValueError("checkpoint changed while loading recovery policy")
            policy.seed_sampling(policy_seed)
        result, _, _ = rt.episode(
            factory,
            work,
            options,
            policy,
            entry["seed"],
            "normal",
            ex.drain_ticks,
            scenario=scenario,
            wall_deadline=deadline,
            partial_on_timeout=True,
            metrics_level="detailed",
            metric_interval=ex.run.metric_interval,
            progress_interval=ex.display.summary_interval,
        )
        result.update(
            candidate=candidate.name,
            load=candidate.load,
            method=entry["method"],
            action_seed=entry["action_seed"],
            fault_index=entry["fault_index"],
            disturbed=entry["disturbed"],
            selection=args.selection,
        )
        result["evaluation_identity"] = {
            "checkpoint_sha256": checkpoint if learning else None,
            "training_progress": getattr(args, "training_progress", None)
            if learning
            else None,
            "method": entry["method"],
            "learning": learning,
            "selection": args.selection,
            "policy_seed": policy_seed,
            "source_sha256": source["implementation_sha256"],
            "runtime_sha256": fingerprint(source["environment"]),
            "algorithm_sha256": manifest["inputs"]["algorithm"],
            "world_index": entry["world_index"],
        }
        if result.get("wall_clock_fragment"):
            rt.write_json(root / "interrupted-case.json", result)
            raise TimeoutError("recovery interrupted; partial evidence retained")
        rt.commit_case(store, index, result)
        rows.append(result)
        rt.publish_cases(root, store)
        tracking.log(
            {
                "phase": "calibration"
                if settings.action == "calibrate"
                else "evaluation",
                "kind": "recovery_case",
                "case_index": index,
                **result,
            }
        )
        rt.publish_progress(
            root,
            {
                "status": "evaluating",
                "phase": "calibration"
                if settings.action == "calibrate"
                else "evaluation",
                "completed_cases": len(rows),
                "total_cases": len(entries),
                "tracking": dict(tracking.status),
            },
        )
    report = {
        "selection": args.selection,
        "schema": "smartsom.recovery-evaluation/v1",
        "definition": definition,
        "definition_sha256": fingerprint(definition),
        "source": source,
        "accepted": {},
        "results": [],
    }
    for candidate in candidates:
        selected = [r for r in rows if r["candidate"] == candidate.name]
        normal = [r for r in selected if not r["disturbed"]]
        platforms = [
            platform(r, candidate, candidate.faults[r["fault_index"]].start_tick)
            for r in normal
        ]
        valid = all(p["status"] == "valid" for p in platforms) and all(
            r["last_arrival"] >= candidate.horizon_ticks for r in normal
        )
        report["accepted"][candidate.name] = valid
        if settings.action == "calibrate":
            report["results"].append(
                {
                    "candidate": candidate.name,
                    "load": candidate.load,
                    "platforms": platforms,
                    "backlog_sustained": all(
                        all(
                            point["arrived_backlog"] > 0
                            for point in r["research_metrics"]["series"]
                            if candidate.faults[r["fault_index"]].start_tick
                            - candidate.window_ticks * candidate.baseline_windows
                            <= point["tick"]
                            <= candidate.faults[r["fault_index"]].start_tick
                        )
                        for r in normal
                    ),
                    "baseline_valid": valid,
                    "mean_throughput": mean(
                        p.get("mean_throughput", 0) for p in platforms
                    ),
                    "saturation_confirmed": False,
                }
            )
        else:
            for row in normal:
                other = next(
                    r
                    for r in selected
                    if r["disturbed"]
                    and all(
                        r[k] == row[k]
                        for k in (
                            "method",
                            "scenario_seed",
                            "action_seed",
                            "fault_index",
                        )
                    )
                )
                report["results"].append(
                    {
                        "candidate": candidate.name,
                        "method": row["method"],
                        "scenario_seed": row["scenario_seed"],
                        "action_seed": row["action_seed"],
                        "fault_index": row["fault_index"],
                        **recovery_evidence(
                            row, other, candidate, candidate.faults[row["fault_index"]]
                        ),
                    }
                )
    if settings.action == "calibrate":
        for result in report["results"]:
            candidate = next(c for c in candidates if c.name == result["candidate"])
            if candidate.load != "saturated":
                continue
            # Compare only candidates differing in supply rate; do not attribute geometry/protocol changes to saturation.
            peers = [
                r
                for r in report["results"]
                if r["baseline_valid"]
                and r["candidate"] != candidate.name
                and any(
                    c.name == r["candidate"]
                    and c.mean_interarrival < candidate.mean_interarrival
                    and c.model_dump(
                        exclude={
                            "name",
                            "mean_interarrival",
                            "orders",
                            "initial_orders",
                        }
                    )
                    == candidate.model_dump(
                        exclude={
                            "name",
                            "mean_interarrival",
                            "orders",
                            "initial_orders",
                        }
                    )
                    for c in candidates
                )
            ]
            confirmed = (
                bool(peers)
                and result["baseline_valid"]
                and result["backlog_sustained"]
                and all(
                    peer["backlog_sustained"]
                    and len(peer["platforms"]) == len(result["platforms"])
                    and all(
                        first.get("mean_throughput", 0) > 0
                        and abs(second["mean_throughput"] - first["mean_throughput"])
                        / first["mean_throughput"]
                        <= settings.maximum_relative_capacity_gain
                        for first, second in zip(
                            result["platforms"], peer["platforms"], strict=True
                        )
                    )
                    for peer in peers
                )
            )
            result["saturation_confirmed"] = confirmed
            report["accepted"][candidate.name] &= confirmed
        rt.write_json(root / "recovery-calibration.json", report)
    rt.write_json(root / "recovery-summary.json", report)
    rt.write_json(root / "summary.json", report)
    rt.publish_progress(
        root,
        {
            "status": "completed",
            "phase": "calibration" if settings.action == "calibrate" else "evaluation",
            "completed_cases": len(rows),
            "total_cases": len(entries),
            "tracking": dict(tracking.status),
        },
    )
    tracking.log(
        {
            "phase": "evaluation",
            "kind": "recovery_summary",
            "selection": args.selection,
            "results": report["results"],
        }
    )
    return report
