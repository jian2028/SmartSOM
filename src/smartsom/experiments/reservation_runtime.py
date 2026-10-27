"""Shared reservation rollout, calibration, training and evaluation runtime.

Run with ``python -m smartsom.experiments.base_diagnostic``. The legacy four-file
CLI retains its existing contracts; these explicitly versioned inputs cannot
silently select the old nonexclusive-intention controller.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

from smartsom.algorithms.base_policy import uses_learning
from smartsom.algorithms.base_rules import BaseRule, SharedMover
from smartsom.config.base_diagnostic import generate_scenario, load_inputs, seed_for
from smartsom.domain.base_decisions import BASE_CONTRACT
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.control import boundary
from smartsom.experiments.evidence import write_json

CONDITIONS = ("normal", "low", "medium", "high")


def identity(paths):
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}


def source_identity():
    from smartsom.experiments.evidence import source_identity as common_identity

    source = common_identity()
    package = Path(__file__).resolve().parents[1]
    root = next((p for p in package.parents if (p / ".git").exists()), None)
    branch = None
    if root is not None:
        try:
            branch = subprocess.check_output(
                ["git", "-C", str(root), "branch", "--show-current"],
                text=True,
                stderr=subprocess.PIPE,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass  # Common source identity records unavailable Git evidence.

    return {
        "commit": source["git"]["commit"],
        "branch": branch,
        "dirty": source["git"]["dirty"],
        "root": str(root or package),
        "environment": source,
        "implementation_sha256": hashlib.sha256(
            b"".join(
                str(p.relative_to(package)).encode() + b"\0" + p.read_bytes()
                for p in sorted(package.rglob("*.py"))
            )
        ).hexdigest(),
    }


def freeze(root, paths, task, extra):
    root = Path(root)
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        current = source_identity()
        from smartsom.experiments.evidence import runtime_source_matches

        if not runtime_source_matches(
            manifest["source"]["environment"], current["environment"]
        ):
            raise ValueError("frozen runtime identity mismatch")
        if (
            manifest["inputs"] != identity(paths)
            or manifest["source"]["implementation_sha256"]
            != current["implementation_sha256"]
            or manifest["source"]["commit"] != current["commit"]
        ):
            raise ValueError("frozen source/input identity mismatch")
        if manifest["task"] != task or any(
            manifest.get(key) != value for key, value in extra.items()
        ):
            raise ValueError("frozen training settings mismatch")
        if task not in {"train", "evaluate", "calibration"}:
            raise ValueError("this task does not support unfinished continuation")
        return manifest
    root.mkdir(parents=True, exist_ok=(root / "plan.json").exists())
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    for name, path in paths.items():
        destination = inputs / (name + ".yaml")
        if destination.resolve() != path.resolve():
            destination.write_bytes(path.read_bytes())
    manifest = {
        "contract": BASE_CONTRACT,
        "task": task,
        "source": source_identity(),
        "inputs": identity(paths),
        "pid": os.getpid(),
        "started_unix": time.time(),
        "evidence": "development-diagnostic",
        **extra,
    }
    write_json(root / "manifest.json", manifest)
    from smartsom.telemetry.runtime import bind

    bind(root)
    return manifest


def metrics(sim, options, condition, seed, elapsed_seconds, decisions):
    n, done = len(sim.demands), len(sim.completed)
    failed = done - len(sim._qualified_shipments)
    complete = done == n
    tardiness = sum(
        max(0, sim.shipment_times[d] - sim.demands[d].due_at) for d in sim.completed
    )
    expected = -(sim._overdue_time + options.quality_penalty * failed) / n
    if not math.isclose(sim.total_reward, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise AssertionError((sim.total_reward, expected))
    if complete and tardiness != sim._overdue_time:
        raise AssertionError("overdue integral did not telescope to final tardiness")
    return {
        "condition": condition,
        "scenario_seed": seed,
        "status": sim.status,
        "termination_reason": "all_delivered" if complete else "drain_deadline",
        "tick": sim.tick,
        "last_arrival": max(d.release_at for d in sim.demands.values()),
        "completed": done,
        "total_orders": n,
        "unfinished": n - done,
        "mean_tardiness": tardiness / n if complete else None,
        "passing_rate": (done - failed) / done if done else None,
        "final_passing_rate": (done - failed) / n if complete else None,
        "joint_cost": -expected if complete else None,
        "observed_return": sim.total_reward,
        "observed_overdue": sim._overdue_time,
        "failed": failed,
        "wall_seconds": elapsed_seconds,
        "decisions": decisions,
        "protocol": dict(sim.protocol.metrics),
        "physics": dict(sim.metrics),
        "final_station_state": {
            m: {
                "status": s["status"],
                "down": s["down"],
                "pre": len(sim.protocol.items(sim.pre[m])),
                "post": len(sim.protocol.items(sim.post[m])),
            }
            for m, s in sim.machine_state.items()
        },
        "final_traffic_state": sim.protocol.traffic_view(),
    }


def episode(
    factory,
    workload,
    options,
    policy,
    seed,
    condition,
    drain,
    *,
    on_progress=None,
    trace=None,
    extend=None,
    wall_deadline=None,
    max_ticks=None,
    diagnostics=False,
    observer=None,
):
    if diagnostics:
        if observer is not None:
            raise ValueError("choose diagnostic counters or an explicit observer")
        from smartsom.experiments.reservation_diagnostics import DecisionCounters

        observer = DecisionCounters()
    sim = ProductionSimulator(
        generate_scenario(factory, workload, seed, condition, drain), contract="base"
    )
    training_cut = max_ticks is not None and max_ticks < sim.scenario.tick_limit
    if training_cut:
        sim.scenario = replace(sim.scenario, tick_limit=max_ticks)
    protocol = sim.protocol
    protocol.configure(options)
    mover = SharedMover(protocol.routes)
    rewards, decisions, history = [], 0, []
    started, last_report = time.monotonic(), time.monotonic()
    last_productive = 0
    needs_critic = getattr(policy, "needs_critic", None)
    training = getattr(policy, "training", False)
    while True:
        boundary()
        if sim.done:
            history.append(
                {
                    "tick": sim.tick,
                    "completed": len(sim.completed),
                    "last_productive": last_productive,
                }
            )
            if sim.status == "truncated" and extend:
                next_drain = extend(sim, history)
                if next_drain is not None:
                    longer = generate_scenario(
                        factory, workload, seed, condition, next_drain
                    )
                    # Same realization continues; faults straddling old horizons
                    # retain their full repair interval, rather than being clipped.
                    assert longer.demands == sim.scenario.demands
                    assert (
                        tuple(
                            o
                            for o in longer.outages
                            if o.start < sim.scenario.tick_limit
                        )
                        == sim.scenario.outages
                    )
                    sim.scenario = longer
                    continue
            break
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            raise TimeoutError("diagnostic wall budget reached")
        begin = sim.tick
        records = []
        generator = protocol.decisions()
        try:
            request = next(generator)
            while True:
                critic = (
                    protocol.critic_features()
                    if (needs_critic(request) if needs_critic else training)
                    else None
                )
                action = policy.choose(request, critic)
                if observer is not None:
                    observer.decision(request, action)
                if trace:
                    records.append({"decision": asdict(request), "action": action})
                decisions += 1
                request = generator.send(action)
        except StopIteration:
            pass
        moves = mover.moves(protocol.traffic_view(), sim.tick)
        outcome = protocol.advance(moves)
        if outcome["rejections"]:
            raise AssertionError(
                f"shared Mover violated core legality: {outcome['rejections']}"
            )
        elapsed = sim.tick - begin
        if observer is not None:
            observer.transition(elapsed, outcome["reward"])
        rewards.extend([0.0] * (elapsed - 1))
        rewards.append(outcome["reward"])
        if any(
            e["kind"] in ("processing_completed", "pickup", "drop_off")
            for e in outcome["events"]
        ):
            last_productive = sim.tick
        if trace:
            # Executed movement/wait intervals permit integer-time reconstruction.
            public_events = [
                {k: v for k, v in e.items() if k not in ("actual_ticks", "defect")}
                for e in outcome["events"]
            ]
            trace.write(
                json.dumps(
                    {
                        "start": begin,
                        "end": sim.tick,
                        "moves": moves,
                        "decisions": records,
                        "events": public_events,
                    }
                )
                + "\n"
            )
        if on_progress and time.monotonic() - last_report >= 10:
            on_progress(
                {
                    "episode_tick": sim.tick,
                    "completed": len(sim.completed),
                    "released": len(sim.released),
                    "decisions": decisions,
                    "last_productive": last_productive,
                }
            )
            last_report = time.monotonic()
    report = metrics(
        sim, options, condition, seed, time.monotonic() - started, decisions
    )
    if training_cut and sim.status != "completed":
        report["termination_reason"] = "training_budget_cut"
    report["horizon_checks"] = history
    if diagnostics:
        report["diagnostics"] = observer.summary()
    return report, rewards, protocol.critic_features() if training else None


def case_store(root, paths, phase, cases, extra):
    from smartsom.config.codec import primitive
    from smartsom.experiments.evaluation_resume import CaseStore

    ex, factory, workload, _, _ = load_inputs(paths["experiment"])
    worlds = {}
    frozen = []
    for index, case in enumerate(cases):
        key = (case[-1], case[1])
        if key not in worlds:
            worlds[key] = primitive(
                generate_scenario(
                    factory,
                    workload,
                    key[0],
                    key[1],
                    extra.get("drain_ticks", ex.drain_ticks),
                )
            )
        frozen.append(
            {
                "case": f"{phase}-{index}",
                "replication": 0,
                "seed": case[-1],
                "selection": list(case[:-1]),
                "scenario": worlds[key],
            }
        )
    return CaseStore(
        root / "case-commits",
        None,
        frozen,
        phase,
        context={"inputs": identity(paths), "phase": phase, **extra},
    )


def commit_case(store, index, result):
    from smartsom.config.codec import digest

    case = store.cases[index]
    store.commit(
        index,
        {
            "case_id": case["case"],
            "replication": 0,
            "seed": case["seed"],
            "world_sha256": digest(case["scenario"]),
            "engineering_failure": False,
            "status": result["status"],
            "payload": result,
        },
    )


def publish_cases(root, store):
    temporary = root / "cases.jsonl.tmp"
    temporary.write_text(
        "".join(json.dumps(store.rows[i]["payload"]) + "\n" for i in sorted(store.rows))
    )
    temporary.replace(root / "cases.jsonl")


def calibrate(args):
    return run_cases(args, _calibrate)


def evaluate(args):
    return run_cases(args, _evaluate)


def run_cases(args, function):
    from smartsom.experiments.evaluation_resume import stage_lock

    # Lock a sibling so the first native/legacy freeze retains its allocation semantics.
    with stage_lock(
        Path(args.output).parent / ("." + Path(args.output).name + "-stage")
    ):
        return function(args)


def _calibrate(args):
    ex, f, w, a, paths = load_inputs(args.experiment)
    a = a.model_copy(update={"method": "rule"})
    root = Path(args.output)
    manifest = freeze(
        root,
        paths,
        "calibration",
        {
            "maximum_seconds": args.seconds,
            "cases": args.cases,
            "max_drain": args.max_drain,
        },
    )
    deadline = time.monotonic() + max(
        0, manifest["started_unix"] + args.seconds - time.time()
    )
    store = case_store(
        root,
        paths,
        "calibration",
        [
            (method, condition, repetition, seed_for(71000, repetition) % (2**31))
            for method in ("rule", "random")
            for condition in CONDITIONS
            for repetition in range(args.cases)
        ],
        {"max_drain": args.max_drain},
    )
    results, frozen_drain, unresolved = [], ex.drain_ticks, False
    # Independent calibration seeds, never reused by validation/test.
    for method in ("rule", "random"):
        for condition in CONDITIONS:
            for repetition in range(args.cases):
                seed = seed_for(71000, repetition) % (2**31)
                index = len(results)
                if index in store.rows:
                    result = store.rows[index]["payload"]
                    results.append(result)
                    if result["status"] == "completed":
                        frozen_drain = max(frozen_drain, result["candidate_drain"])
                    unresolved = unresolved or result.get("unresolved", False)
                    continue
                current = ex.drain_ticks
                case_unresolved = False

                def extend(sim, history):
                    nonlocal current, case_unresolved
                    recent = sim.tick - history[-1]["last_productive"] < 1000
                    if not recent:
                        return None  # diagnose stalled policy, not a longer time limit
                    if time.monotonic() >= deadline or current >= args.max_drain:
                        case_unresolved = True
                        return None
                    current += 2000
                    return current

                try:
                    result, _, _ = episode(
                        f,
                        w,
                        a,
                        BaseRule(method, seed_for(seed, method)),
                        seed,
                        condition,
                        ex.drain_ticks,
                        extend=extend,
                        wall_deadline=deadline,
                        diagnostics=ex.run.diagnostics,
                    )
                except TimeoutError:
                    unresolved = True
                    break
                result["method"] = method
                result["candidate_drain"] = current
                result["stalled"] = (
                    result["status"] == "truncated"
                    and result["tick"] - result["horizon_checks"][-1]["last_productive"]
                    >= 1000
                )
                result["unresolved"] = case_unresolved
                unresolved = unresolved or case_unresolved
                commit_case(store, index, result)
                results.append(result)
                if result["status"] == "completed":
                    frozen_drain = max(frozen_drain, current)
                write_json(
                    root / "progress.json",
                    {
                        "status": "calibrating",
                        "cases": len(results),
                        "candidate_drain": frozen_drain,
                    },
                )
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "method",
                                "condition",
                                "status",
                                "tick",
                                "completed",
                                "wall_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
    # All conditions must have at least one completed rule check; stalled random
    # runs remain failures and cannot be hidden by successful-subset averaging.
    covered = {
        r["condition"]
        for r in results
        if r["method"] == "rule" and r["status"] == "completed"
    }
    frozen = (
        not unresolved and covered == set(CONDITIONS) and len(results) == 8 * args.cases
    )
    report = {
        "status": "frozen" if frozen else "calibration_incomplete",
        "drain_ticks": frozen_drain,
        "inputs": identity(paths),
        "results": results,
        "source": source_identity(),
        "notes": "stalled cases are policy/protocol diagnostics, not justification for unlimited extension",
    }
    if getattr(args, "native", False):
        from smartsom.config.reservation import calibration_identity

        report["calibration_identity"] = calibration_identity(paths)
    write_json(root / "calibration.json", report)
    write_json(
        root / "progress.json",
        {
            "status": report["status"],
            "drain_ticks": frozen_drain,
            "cases": len(results),
        },
    )
    if not frozen:
        raise RuntimeError("calibration incomplete; training launch refused")


def frozen_drain(args, paths):
    report = json.loads(Path(args.calibration).read_text())
    matches = report["inputs"] == identity(paths)
    if getattr(args, "native", False):
        from smartsom.config.reservation import calibration_identity

        matches = report.get("calibration_identity") == calibration_identity(paths)
    if report["status"] != "frozen" or not matches:
        raise ValueError("matching completed calibration required before comparison")
    return report["drain_ticks"]


def train(args):
    from smartsom.experiments.reservation_session import train as execute

    return execute(args)


def _evaluate(args):
    ex, f, w, a, paths = load_inputs(args.experiment)
    drain = frozen_drain(args, paths)
    root = Path(args.output)
    methods = args.method or ["rule", "random"]
    manifest = freeze(
        root,
        paths,
        "evaluate",
        {
            "drain_ticks": drain,
            "methods": methods,
            "checkpoint": args.checkpoint,
            "selection": args.selection,
            "split": args.split,
            "cases": args.cases,
            "action_seeds": args.action_seeds,
            "maximum_seconds": getattr(args, "seconds", None),
            "deadline_unix": getattr(args, "deadline_unix", None),
        },
    )
    count = args.cases or (
        ex.test_cases if args.split == "test" else ex.validation_cases
    )
    if count % 4:
        raise ValueError("evaluation cases must be balanced across four conditions")
    results = []
    deadline = (
        time.monotonic() + max(0, manifest["started_unix"] + args.seconds - time.time())
        if getattr(args, "native", False)
        else None
    )
    if getattr(args, "deadline_unix", None) is not None:
        deadline = min(
            deadline, time.monotonic() + max(0, args.deadline_unix - time.time())
        )
    store = case_store(
        root,
        paths,
        "evaluate",
        [
            (
                method,
                CONDITIONS[index % 4],
                action_seed,
                seed_for(80000 if args.split == "validation" else 90000, index // 4)
                % (2**31),
            )
            for method in methods
            for index in range(count)
            for action_seed in range(args.action_seeds)
        ],
        {
            "selection": args.selection,
            "drain_ticks": drain,
            "checkpoint_sha256": hashlib.sha256(
                Path(args.checkpoint).read_bytes()
            ).hexdigest()
            if args.checkpoint
            else None,
        },
    )
    for method in methods:
        for index in range(count):
            condition = CONDITIONS[index % 4]
            seed = seed_for(
                80000 if args.split == "validation" else 90000, index // 4
            ) % (2**31)
            for action_seed in range(args.action_seeds):
                case_index = len(results)
                if case_index in store.rows:
                    results.append(store.rows[case_index]["payload"])
                    continue
                from smartsom.algorithms.base_policy import build_policy

                scenario_options = (
                    a
                    if method == "configured"
                    else a.model_copy(
                        update={"method": method, "bindings": {}, "providers": {}}
                    )
                )
                learning = uses_learning(scenario_options)
                policy_seed = seed_for(seed, action_seed)
                policy = build_policy(
                    scenario_options,
                    policy_seed % (2**31) if learning else policy_seed,
                    training=False,
                    selection=args.selection,
                    numerical_threads=getattr(args, "numerical_threads", 1),
                )
                if learning:
                    policy.load(args.checkpoint)
                    learner = getattr(policy, "learner", policy)
                    learner.generator.manual_seed(policy_seed % (2**31))
                recording = ex.record_first_case and index == 0 and action_seed == 0
                with (
                    (root / f"trace-{method}.jsonl").open("w")
                    if recording
                    else nullcontext(None)
                ) as trace:
                    result, _, _ = episode(
                        f,
                        w,
                        scenario_options,
                        policy,
                        seed,
                        condition,
                        drain,
                        trace=trace,
                        wall_deadline=deadline,
                        diagnostics=ex.run.diagnostics,
                    )
                result.update(
                    method=method, action_seed=action_seed, selection=args.selection
                )
                commit_case(store, case_index, result)
                results.append(result)
                publish_cases(root, store)
                write_json(
                    root / "progress.json",
                    {
                        "status": "evaluating",
                        "cases": len(results),
                        "total": len(methods) * count * args.action_seeds,
                    },
                )
    publish_cases(root, store)
    summary = {}
    for method in methods:
        summary[method] = {}
        for condition in (*CONDITIONS, "all"):
            rows = [
                r
                for r in results
                if r["method"] == method
                and (condition == "all" or r["condition"] == condition)
            ]
            complete = [r for r in rows if r["joint_cost"] is not None]
            summary[method][condition] = {
                "cases": len(rows),
                "not_drained_rate": 1 - len(complete) / len(rows),
                "mean_tardiness": sum(r["mean_tardiness"] for r in complete) / len(rows)
                if len(complete) == len(rows)
                else None,
                "joint_cost": sum(r["joint_cost"] for r in complete) / len(rows)
                if len(complete) == len(rows)
                else None,
                "passing_rate": sum(r["failed"] for r in rows)
                / sum(r["completed"] for r in rows)
                if sum(r["completed"] for r in rows)
                else None,
            }
            if summary[method][condition]["passing_rate"] is not None:
                summary[method][condition]["passing_rate"] = (
                    1 - summary[method][condition]["passing_rate"]
                )
            summary[method][condition]["observed_passing_rate"] = summary[method][
                condition
            ]["passing_rate"]
            if len(complete) != len(rows):
                summary[method][condition]["passing_rate"] = None
    write_json(root / "summary.json", summary)
    write_json(root / "progress.json", {"status": "completed", "cases": len(results)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("experiment")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("experiment")
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--seconds", type=float, default=3600)
    calibration.add_argument("--cases", type=int, default=2)
    calibration.add_argument("--max-drain", type=int, default=10000)
    for name in ("train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("experiment")
        sub.add_argument("--output", required=True)
        sub.add_argument("--calibration", required=True)
        if name == "train":
            sub.add_argument("--seed", type=int, default=101)
            sub.add_argument("--ticks", type=int)
            sub.add_argument("--seconds", type=float, default=14400)
        else:
            sub.add_argument(
                "--method", action="append", choices=("ppo", "rule", "random")
            )
            sub.add_argument("--checkpoint")
            sub.add_argument("--cases", type=int)
            sub.add_argument(
                "--split", choices=("validation", "test"), default="validation"
            )
            sub.add_argument(
                "--selection", choices=("sample", "argmax"), default="sample"
            )
            sub.add_argument("--action-seeds", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "check":
        ex, f, w, a, _ = load_inputs(args.experiment)
        scenario = generate_scenario(f, w, 0, "high", ex.drain_ticks)
        sim = ProductionSimulator(scenario, contract="base")
        reachable = sim.protocol.routes.distances(
            (tuple(next(iter(sim.agvs.values()))["cell"]),)
        )
        assert all(
            c in reachable
            for ports in sim.protocol.routes.ports.values()
            for c in ports
        )
        print(
            json.dumps(
                {
                    "contract": BASE_CONTRACT,
                    "machines": len(f.machines),
                    "vehicles": len(f.agvs),
                    "orders": len(scenario.demands),
                    "last_arrival": scenario.demands[-1].release_at,
                    "deadline": scenario.tick_limit,
                    "drain_frozen": ex.drain_frozen,
                }
            )
        )
    else:
        try:
            {"calibrate": calibrate, "train": train, "evaluate": evaluate}[
                args.command
            ](args)
        except BaseException as exc:
            root = Path(args.output)
            manifest = root / "manifest.json"
            if (
                manifest.is_file()
                and json.loads(manifest.read_text()).get("pid") == os.getpid()
            ):
                write_json(
                    root / "failure.json",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "pid": os.getpid(),
                    },
                )
            raise


if __name__ == "__main__":
    main()
