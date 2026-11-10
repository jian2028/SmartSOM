"""Shared current rollout, calibration, training and evaluation runtime."""

import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from smartsom.algorithms.base_policy import (
    build_policy,
    evaluation_recipe,
    uses_learning,
)
from smartsom.config.base_diagnostic import generate_scenario, load_inputs, seed_for
from smartsom.domain.base_decisions import BASE_CONTRACT
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.control import boundary
from smartsom.experiments.evidence import write_json
from smartsom.experiments.resource_guard import check as check_resources
from smartsom.experiments.resource_guard import guarded
from smartsom.traffic import build_mover

CONDITIONS = ("normal", "low", "medium", "high")


def publish_progress(root, payload, *, display_root=None):
    """Publish local/terminal progress without uploading partial case evidence."""
    from smartsom.telemetry.current_progress import presentation
    from smartsom.telemetry.runtime import emit

    write_json(root / "progress.json", payload)
    display_root = Path(display_root or root)
    if display_root != root:
        previous = display_root / "progress.json"
        payload = {
            **(json.loads(previous.read_text()) if previous.exists() else {}),
            **payload,
            "status": "evaluating",
            "stage": payload["phase"],
            "panel_status": payload["status"],
        }
        write_json(previous, payload)
    event, total, unit = presentation(payload)
    emit(
        display_root,
        event,
        total=total,
        unit=unit,
        phase=payload.get("phase"),
        final=payload.get("status") in {"completed", "frozen", "incomplete"},
    )


def identity(paths):
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}


def prepared_identity(paths):
    """Bind detached parsed inputs to both science and operational overrides."""
    result = identity(paths)
    experiment = paths["experiment"]
    execution = experiment.parent.parent / "execution.json"
    if experiment.parent.name == "inputs":
        result["execution"] = (
            hashlib.sha256(execution.read_bytes()).hexdigest()
            if execution.exists()
            else None
        )
    return result


def source_identity():
    from smartsom.experiments.evidence import source_identity as common_identity

    source = common_identity()
    package = Path(__file__).resolve().parents[1]
    root = next((p for p in package.parents if (p / ".git").exists()), None)
    # Every path below is yielded beneath package; slicing preserves its exact
    # relative spelling without repeatedly walking this deep absolute parent chain.
    prefix = len(str(package)) + 1
    return {
        "commit": source["git"]["commit"],
        "branch": source["git"].get("branch"),
        "dirty": source["git"]["dirty"],
        "root": str(root or package),
        "environment": source,
        "implementation_sha256": hashlib.sha256(
            b"".join(
                str(p)[prefix:].encode() + b"\0" + p.read_bytes()
                for p in sorted(package.rglob("*.py"))
            )
        ).hexdigest(),
    }


def freeze(root, paths, task, extra):
    root = Path(root)
    launch_plan = (
        json.loads((root / "plan.json").read_text())
        if (root / "plan.json").exists()
        else {}
    )
    extra = dict(extra)
    for field in ("tracking_group", "experiment_short_name"):
        if extra.get(field) is None and launch_plan.get(field) is not None:
            extra[field] = launch_plan[field]
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
        history = manifest.get("execution_history", [])
        platform = current["environment"]["platform"]
        previous = (
            history[-1]["platform"]
            if history
            else manifest["source"]["environment"]["platform"]
        )
        if previous != platform:
            manifest["execution_history"] = history + [
                {"platform": platform, "time": time.time()}
            ]
            write_json(root / "manifest.json", manifest)
        return manifest
    root.mkdir(parents=True, exist_ok=(root / "plan.json").exists())
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    for name, path in paths.items():
        destination = inputs / (name + ".yaml")
        if destination.resolve() != path.resolve():
            destination.write_bytes(path.read_bytes())
    from uuid import uuid4

    from smartsom.config.codec import read_document
    from smartsom.config.launch import scientific_identity

    manifest = {
        "run_id": launch_plan.get("run_id") or uuid4().hex,
        "runtime_contract": "smartsom.current-runtime/v1",
        "scientific_sha256": scientific_identity(
            {k: read_document(p) for k, p in paths.items()}
        ),
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


def episode(*args, replay_directory=None, replay_source=None, **kwargs):
    if replay_directory is None:
        return _episode(*args, **kwargs)
    from smartsom.trace.current import CurrentRecording

    parent = Path(replay_directory)
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        directory = parent / f"attempt-{attempt:06d}"
        try:
            directory.mkdir()
            break
        except FileExistsError:
            attempt += 1
    with (
        (directory / "decisions.jsonl").open("w") as trace,
        CurrentRecording(directory / "replay", source=replay_source) as recording,
    ):
        report, rewards, critic = _episode(
            *args, trace=trace, recording=recording, **kwargs
        )
    report["recording_directory"] = str(directory)
    return report, rewards, critic


def _episode(
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
    recording=None,
    extend=None,
    wall_deadline=None,
    max_ticks=None,
    diagnostics=False,
    observer=None,
    metrics_level="auto",
    metric_interval=50,
    window_analysis=None,
    scenario=None,
    partial_on_timeout=False,
    common_window_end=None,
    progress_interval=5,
    control_check=None,
    on_fragment=None,
    fragment_ticks=256,
    episode_end="drain_deadline",
):
    if episode_end not in {"drain_deadline", "all_delivered"}:
        raise ValueError("unknown episode end mode")
    if episode_end == "all_delivered" and extend is not None:
        raise ValueError("all_delivered cannot combine with a custom horizon extension")
    if diagnostics:
        if observer is not None:
            raise ValueError("choose diagnostic counters or an explicit observer")
        from smartsom.experiments.reservation_diagnostics import DecisionCounters

        observer = DecisionCounters()
    sim = ProductionSimulator(
        scenario or generate_scenario(factory, workload, seed, condition, drain),
        contract="base",
    )
    training_cut = max_ticks is not None and (
        max_ticks < sim.scenario.tick_limit
        or (episode_end == "all_delivered" and max_ticks == sim.scenario.tick_limit)
    )
    if training_cut:
        sim.scenario = replace(sim.scenario, tick_limit=max_ticks)
    protocol = sim.protocol
    protocol.configure(options)
    if options.control == "centralized":
        from smartsom.engine.central_control import CentralControl

        CentralControl(sim)
    from smartsom.algorithms.observation import build_observation

    protocol.observation_transform = (
        policy.observation_transform
        if hasattr(policy, "observation_transform")
        else build_observation(options)
    )
    begin_episode = getattr(policy, "begin_episode", None)
    if begin_episode is not None:
        begin_episode()
    if recording is not None:
        recording.begin(sim)
        recording.recorder.manifest["learning_components"] = options.science_identity()
    mover = (
        policy.create_mover(protocol.routes, seed)
        if hasattr(policy, "create_mover")
        else build_mover(options, protocol.routes, seed)
    )
    from smartsom.algorithms.training import TrainingComponents
    from smartsom.domain.training import CommittedAction

    training_components = getattr(
        policy, "training_components", None
    ) or TrainingComponents(options)
    training_components.begin(protocol)
    collect_reward_actions = getattr(
        training_components.reward, "requires_actions", True
    )
    rewards, decisions, history = training_components.timeline, 0, []
    from smartsom.algorithms.rule_actions import RuleMetrics

    rule_metrics = RuleMetrics()
    started, last_report = time.monotonic(), time.monotonic()
    last_productive = 0
    needs_critic = getattr(policy, "needs_critic", None)
    training = getattr(policy, "training", False)
    if metrics_level not in {"auto", "off", "summary", "detailed"}:
        raise ValueError("unknown research metrics level")
    collector = None
    if metrics_level != "off":
        from smartsom.telemetry.research_metrics import ResearchMetrics

        collector = ResearchMetrics(
            sim,
            detailed=metrics_level == "detailed"
            or (metrics_level == "auto" and not training),
            interval=metric_interval,
            trip_sink=recording.append_trip if recording is not None else None,
        )
    wall_fragment = False
    next_fragment = fragment_ticks
    while True:
        boundary()
        if control_check is not None:
            control_check()
        if sim.done:
            break
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            if partial_on_timeout:
                wall_fragment = True
                break
            raise TimeoutError("diagnostic wall budget reached")
        begin = sim.tick
        committed_actions = []
        records = []
        generator = protocol.decisions()
        try:
            request = next(generator)
            while True:
                critic = (
                    training_components.observe_critic(protocol, request)
                    if (needs_critic(request) if needs_critic else training)
                    else None
                )
                action = policy.choose(request, critic)
                if collect_reward_actions:
                    selected = next(
                        (
                            c
                            for c in request.candidates
                            if c.identity == getattr(policy, "selected_identity", None)
                            or (
                                getattr(policy, "selected_identity", None) is None
                                and (c.action == action or c.identity == action)
                            )
                        ),
                        None,
                    )
                    if selected is None:
                        raise ValueError("policy returned an illegal candidate action")
                    committed_actions.append(
                        CommittedAction(
                            request.tick,
                            request.group,
                            request.owner,
                            request.stage,
                            selected.identity,
                            action,
                        )
                    )
                if recording is not None:
                    recording.decision(
                        request,
                        action,
                        selected_identity=(
                            getattr(policy, "selected_identity", None)
                            if not options.action_representation.merge_aliases
                            else None
                        ),
                    )
                rule_metrics.add(
                    request,
                    action,
                    selected_identity=getattr(policy, "selected_identity", None),
                )
                if observer is not None:
                    observer.decision(request, action)
                if trace:
                    from smartsom.learning.records import recorded_request

                    records.append(
                        {
                            "decision": recorded_request(request),
                            "action": action,
                            **(
                                {
                                    "selected_identity": getattr(
                                        policy, "selected_identity", None
                                    )
                                }
                                if not options.action_representation.merge_aliases
                                else {}
                            ),
                        }
                    )
                if collector is not None:
                    collector.decision(request, action)
                decisions += 1
                request = generator.send(action)
        except StopIteration:
            pass
        moves = mover.moves(protocol.traffic_view(), sim.tick)
        while (
            episode_end == "all_delivered"
            and not training_cut
            and len(sim.completed) < len(sim.demands)
            and sim.tick + protocol.idle_interval(moves) >= sim.scenario.tick_limit
        ):
            # Extend before a physical transition can publish a false task terminal.
            next_drain = (
                sim.scenario.tick_limit
                - max(d.release_at for d in sim.demands.values())
                + drain
            )
            longer = generate_scenario(factory, workload, seed, condition, next_drain)
            if (
                replace(
                    longer,
                    tick_limit=sim.scenario.tick_limit,
                    outages=sim.scenario.outages,
                )
                != sim.scenario
                or tuple(o for o in longer.outages if o.start < sim.scenario.tick_limit)
                != sim.scenario.outages
            ):
                raise ValueError(
                    "completion extension changed the frozen physical world"
                )
            if max_ticks is not None and longer.tick_limit >= max_ticks:
                longer = replace(longer, tick_limit=max_ticks)
                training_cut = True
            history.append({"tick": sim.tick, "extended_to": longer.tick_limit})
            sim.scenario = longer
            if recording is not None:
                from smartsom.config.codec import primitive

                recording.recorder.manifest["inputs"]["scenario"] = primitive(longer)
        if collector is not None:
            collector.before()
        reward_before = training_components.snapshot(protocol)
        outcome = protocol.advance(
            moves, on_tick=recording.append if recording is not None else None
        )
        if collector is not None:
            collector.after(outcome)
        if outcome["rejections"]:
            raise AssertionError(
                f"shared Mover violated core legality: {outcome['rejections']}"
            )
        elapsed = sim.tick - begin
        if observer is not None:
            observer.transition(elapsed, outcome["reward"])
        if any(
            e["kind"] in ("processing_completed", "pickup", "drop_off")
            for e in outcome["events"]
        ):
            last_productive = sim.tick
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
        reward_result = training_components.transition(
            protocol,
            begin,
            reward_before,
            committed_actions,
            outcome,
            end_reason=(
                "all_delivered"
                if sim.status == "completed"
                else "training_budget_cut"
                if sim.done and training_cut
                else "drain_deadline"
                if sim.status == "truncated"
                else None
            ),
        )
        if recording is not None:
            recording.reward(begin, sim.tick, reward_result)
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
        if on_fragment is not None and not sim.done and sim.tick >= next_fragment:
            critic = training_components.finish(protocol, policy)
            on_fragment(policy, rewards, critic)
            next_fragment = sim.tick + fragment_ticks
        if on_progress and time.monotonic() - last_report >= progress_interval:
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
    if wall_fragment:
        report.update(
            status="interrupted",
            termination_reason="wall_clock_interrupted",
            wall_clock_fragment=True,
        )
    if common_window_end is not None:
        observable = sim.tick >= common_window_end or (
            sim.status == "completed" and not wall_fragment
        )
        delivered = sum(t <= common_window_end for t in sim.shipment_times.values())
        report["common_window"] = {
            "start_tick": 0,
            "end_tick": common_window_end,
            "status": "observed" if observable else "censored",
            "completed": delivered if observable else None,
            "throughput": delivered / common_window_end if observable else None,
        }
    report["learning_return"] = sum(rewards)
    report["learning_cost"] = (
        -report["learning_return"] if report["status"] == "completed" else None
    )
    report["makespan"] = sim.tick if report["status"] == "completed" else None
    report["completion_tail_ticks"] = (
        sim.tick - report["last_arrival"] if report["status"] == "completed" else None
    )
    report["total_tardiness"] = (
        report["observed_overdue"] if report["status"] == "completed" else None
    )
    report["episode_end"] = episode_end
    report["decoder"] = getattr(getattr(policy, "learner", policy), "decoder", None)
    component_metrics = getattr(policy, "episode_metrics", lambda: {})()
    if component_metrics:
        report["policy_metrics"] = component_metrics
    report["rule_actions"] = rule_metrics.summary()
    report["horizon_checks"] = history
    if collector is not None:
        report["research_metrics"] = collector.finish(report)
    if window_analysis is not None:
        from smartsom.telemetry.window_series import throughput_series

        report["window_analysis"] = throughput_series(report, window_analysis)
        report["window_analysis"]["recovery_window_ticks"] = (
            window_analysis.recovery_window_ticks
        )
        if window_analysis.same_run is not None:
            from smartsom.telemetry.paired_export import same_run_episode

            report["window_analysis"]["same_run_recovery"] = same_run_episode(
                report, window_analysis, metric_interval
            )
    if diagnostics:
        report["diagnostics"] = observer.summary()
    rewards.end_reason = report["termination_reason"]
    final_critic = training_components.finish(protocol, policy)
    report["role_returns"] = {k: sum(v) for k, v in rewards.roles.items()}
    report["agent_returns"] = {
        f"{g}/{o}": sum(v) for (g, o), v in rewards.agents.items()
    }
    return report, rewards, final_critic if training else None


def case_store(root, paths, phase, cases, extra, *, scenarios=None):
    from smartsom.config.codec import primitive
    from smartsom.experiments.evaluation_resume import CaseStore

    ex, factory, workload, _, _ = load_inputs(paths["experiment"])
    worlds = {}
    frozen = []
    for index, case in enumerate(cases):
        key = (case[-1], case[1])
        if key not in worlds:
            worlds[key] = (
                primitive(scenarios[index])
                if scenarios is not None
                else primitive(
                    generate_scenario(
                        *ex.inputs_for(key[1], factory, workload),
                        key[0],
                        key[1],
                        extra.get("drain_ticks", ex.drain_ticks),
                    )
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

    check_resources("case_commit", force=True)
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
    check_resources("case_committed", force=True)


def publish_cases(root, store):
    temporary = root / "cases.jsonl.tmp"
    temporary.write_text(
        "".join(json.dumps(store.rows[i]["payload"]) + "\n" for i in sorted(store.rows))
    )
    temporary.replace(root / "cases.jsonl")


def calibrate(args):
    ex, *_ = load_inputs(args.experiment)
    if ex.evaluation.recovery.action != "disabled":
        return evaluate(args)
    return run_cases(args, _calibrate)


def evaluate(args):
    args._tracking = None
    try:
        result = run_cases(args, _evaluate)
    except BaseException:
        if args._tracking is not None:
            args._tracking.close(failed=True) if not getattr(
                args, "tracking", None
            ) else None
        raise
    else:
        if args._tracking is not None:
            args._tracking.close() if not getattr(args, "tracking", None) else None
        return result


@guarded
def run_cases(args, function):
    from smartsom.experiments.evaluation_resume import stage_lock

    # Lock a sibling so the first freeze retains its allocation semantics.
    with stage_lock(
        Path(args.output).parent / ("." + Path(args.output).name + "-stage")
    ):
        return function(args)


def _calibrate(args):
    ex, f, w, a, paths = load_inputs(args.experiment)
    # Legacy commands supply this limit outside the typed Experiment document.
    if args.max_drain < ex.drain_ticks:
        raise ValueError("max_drain must be at least the initial drain_ticks")
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
    # Independent calibration seeds, never reused by periodic_evaluation/test.
    for method in ("rule", "random"):
        baseline = evaluation_recipe(a, method)
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
                    current = min(current + 2000, args.max_drain)
                    return current

                try:
                    check_resources("calibration_model_build", force=True)
                    policy = build_policy(
                        baseline, seed_for(seed, method), training=False
                    )
                    check_resources("calibration_rollout", force=True)
                    result, _, _ = episode(
                        *ex.inputs_for(condition, f, w),
                        baseline,
                        policy,
                        seed,
                        condition,
                        ex.drain_ticks,
                        extend=extend,
                        wall_deadline=deadline,
                        diagnostics=ex.run.diagnostics,
                        metrics_level=ex.run.metrics,
                        metric_interval=ex.run.metric_interval,
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
                publish_progress(
                    root,
                    {
                        "status": "calibrating",
                        "phase": "calibration",
                        "completed_cases": len(results),
                        "total_cases": len(store.cases),
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
    publish_progress(
        root,
        {
            "status": report["status"],
            "phase": "calibration",
            "completed_cases": len(results),
            "total_cases": len(store.cases),
            "drain_ticks": frozen_drain,
            "cases": len(results),
        },
    )
    if not frozen:
        raise RuntimeError("calibration incomplete; training launch refused")


def frozen_drain(args, paths):
    ex, *_ = load_inputs(paths["experiment"])
    if not args.calibration and ex.drain_frozen:
        return ex.drain_ticks
    report = json.loads(Path(args.calibration).read_text())
    matches = report["inputs"] == identity(paths)
    if getattr(args, "native", False):
        from smartsom.config.reservation import calibration_identity

        matches = report.get("calibration_identity") == calibration_identity(paths)
    if report["status"] != "frozen" or not matches:
        raise ValueError("matching completed calibration required before comparison")
    return report["drain_ticks"]


@guarded
def train(args):
    from smartsom.experiments.reservation_session import train as execute

    return execute(args)


def checkpoint_training_evidence(args):
    """Verify optional committed training evidence for every evaluation route."""
    checkpoint_digest = (
        hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if getattr(args, "checkpoint", None)
        else None
    )
    supplied_context = getattr(args, "training_context", None)
    args.training_context = None
    if checkpoint_digest:
        sidecar = Path(args.checkpoint).with_suffix(".pt.json")
        if sidecar.exists():
            evidence = json.loads(sidecar.read_text())
            if evidence.get("schema") != "smartsom.checkpoint-evidence/v1":
                raise ValueError("unsupported checkpoint training evidence schema")
            if evidence.get("checkpoint_sha256") != checkpoint_digest:
                raise ValueError("checkpoint training evidence digest mismatch")
            if (
                evidence.get("source_sha256")
                != source_identity()["implementation_sha256"]
            ):
                raise ValueError(
                    "checkpoint training source differs; use its original source"
                )
            if (
                any(
                    type(evidence.get(key)) is not int or evidence[key] < 0
                    for key in ("episodes", "physical_ticks")
                )
                or type(evidence.get("train_seed")) is not int
                or not evidence.get("run_id")
            ):
                raise ValueError("incomplete checkpoint training evidence")
            saved_progress = {
                key: evidence[key]
                for key in (
                    "train_seed",
                    "episodes",
                    "physical_ticks",
                    "source_sha256",
                    "run_id",
                )
            }
            from smartsom.telemetry.evaluation_summary import validate_training_context

            args.training_context = validate_training_context(
                evidence.get("training_context")
            )
            supplied = getattr(args, "training_progress", None) or {}
            if any(
                key in supplied and supplied[key] != value
                for key, value in saved_progress.items()
            ):
                raise ValueError(
                    "evaluation training progress differs from checkpoint evidence"
                )
            args.training_progress = {**supplied, **saved_progress}
    if supplied_context is not None and supplied_context != args.training_context:
        raise ValueError("evaluation training context differs from checkpoint evidence")
    return checkpoint_digest


def evaluate_case(settings, task, prepared, *, on_progress=None, control_check=None):
    """One independent case; publication and receipts belong to the parent."""
    from types import SimpleNamespace

    ex, f, w, a, _ = prepared
    args = SimpleNamespace(**settings)
    method, condition, action_seed, seed = task["case"]
    actual_drain = settings["drain"]
    deadline = settings["deadline"]
    checkpoint_digest = settings["checkpoint_digest"]
    protocol = ex.evaluation
    drain = settings["original_drain"]
    imported = task["imported"]
    from smartsom.algorithms.base_policy import build_policy

    scenario_options = evaluation_recipe(a, method)
    learning = uses_learning(scenario_options)
    policy_seed = action_seed if task["explicit_seed"] else seed_for(seed, action_seed)
    sampling_seed = (
        policy_seed if task["explicit_seed"] or not learning else policy_seed % (2**31)
    )
    check_resources("evaluation_model_build", force=True)
    policy = build_policy(
        scenario_options,
        sampling_seed,
        training=False,
        selection=args.selection,
        numerical_threads=getattr(args, "numerical_threads", 1),
    )
    check_resources("evaluation_model_built", force=True)
    if learning:
        if (
            hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
            != checkpoint_digest
        ):
            raise ValueError("evaluation checkpoint changed during panel")
        check_resources("evaluation_model_load", force=True)
        policy.load(args.checkpoint)
        check_resources("evaluation_model_loaded", force=True)
        if (
            hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
            != checkpoint_digest
        ):
            raise ValueError("evaluation checkpoint changed during load")
        policy.seed_sampling(sampling_seed)
    replay_directory = task["replay_directory"]
    case_deadline = deadline
    case_limited = False
    if args.selection in protocol.case_seconds:
        limit = time.monotonic() + protocol.case_seconds[args.selection]
        case_limited = deadline is None or limit < deadline
        if case_limited:
            case_deadline = limit
    check_resources("evaluation_rollout", force=True)
    result, _, _ = episode(
        *ex.inputs_for(condition, f, w),
        scenario_options,
        policy,
        seed,
        condition,
        actual_drain,
        scenario=task["scenario"],
        replay_directory=replay_directory,
        replay_source=settings["source"],
        wall_deadline=case_deadline,
        diagnostics=ex.run.diagnostics,
        metrics_level=ex.run.metrics,
        metric_interval=ex.run.metric_interval,
        window_analysis=ex.evaluation.window_analysis,
        partial_on_timeout=True,
        common_window_end=protocol.common_window_end_tick
        or (
            task["scenario"].tick_limit
            if imported
            else generate_scenario(
                *ex.inputs_for(condition, f, w), seed, condition, drain
            ).tick_limit
        ),
        progress_interval=ex.display.summary_interval,
        on_progress=on_progress,
        control_check=control_check,
        episode_end=ex.episode_end,
    )
    return result, learning, sampling_seed, case_limited


def _evaluate(args):
    ex, f, w, a, paths = load_inputs(args.experiment)
    if args.split in {"validation", "periodic_evaluation"}:
        args.split = ex.validation_split
    ex.validate_execution()
    if ex.evaluation.recovery.action != "disabled":
        from smartsom.experiments.recovery_evaluation import run

        return run(args)
    drain = frozen_drain(args, paths)
    protocol = ex.evaluation
    root = Path(args.output)
    methods = args.method or ["rule", "random"]
    from smartsom.telemetry.research_metrics import fingerprint

    environment = source_identity()["environment"]
    runtime_digest = fingerprint(
        {
            "python": environment["python"],
            "packages": environment["packages"],
            "numerical_threads": getattr(args, "numerical_threads", 1),
        }
    )
    checkpoint_digest = checkpoint_training_evidence(args)
    manifest = freeze(
        root,
        paths,
        "evaluate",
        {
            "drain_ticks": drain,
            "evaluation_protocol": protocol.model_dump(mode="json", by_alias=True),
            "tracking_group": getattr(args, "tracking_group", None),
            "methods": methods,
            "checkpoint": Path(args.checkpoint).name if args.checkpoint else None,
            "checkpoint_sha256": checkpoint_digest,
            "selection": args.selection,
            "split": args.split,
            "cases": args.cases,
            "action_seeds": args.action_seeds,
            "evaluation_runtime_sha256": runtime_digest,
            "training_progress": getattr(args, "training_progress", None),
            "training_context": getattr(args, "training_context", None),
            "training_seed": (getattr(args, "training_progress", None) or {}).get(
                "train_seed"
            ),
            "maximum_seconds": getattr(args, "seconds", None),
            "deadline_unix": getattr(args, "deadline_unix", None),
        },
    )
    from smartsom.telemetry.experiment_tracking import ExperimentTracking

    args._tracking = getattr(args, "tracking", None)
    if args._tracking is None:
        args._tracking = ExperimentTracking(root, ex.tracking, job_type="evaluate")
        args._tracking.start(manifest, ex.model_dump(mode="json", by_alias=True))
    count = args.cases or (
        ex.test_cases if args.split == "test" else ex.validation_cases
    )
    from smartsom.config.evaluation_cases import experiment_panels

    actual_drain = (
        math.ceil(drain * protocol.completion_drain_multiplier)
        if protocol.mode == "completion"
        else drain
    )
    imported_panels = experiment_panels(
        ex, Path(args.experiment).parent, f, w, drain_ticks=actual_drain
    )
    imported = imported_panels.get(args.split)
    if imported is not None:
        if args.split in ex.case_set_sha256:
            from smartsom.config.codec import digest

            if digest(imported) != ex.case_set_sha256[args.split]:
                raise ValueError(
                    "fixed case inventory digest differs from the frozen experiment"
                )
        if count != len(imported):
            raise ValueError("evaluation count differs from frozen case inventory")
    elif count % 4:
        raise ValueError("evaluation cases must be balanced across four conditions")
    results = []
    deadline = (
        time.monotonic() + max(0, manifest["started_unix"] + args.seconds - time.time())
        if getattr(args, "native", False) and args.seconds is not None
        else None
    )
    if getattr(args, "deadline_unix", None) is not None:
        absolute_deadline = time.monotonic() + max(0, args.deadline_unix - time.time())
        deadline = (
            min(deadline, absolute_deadline)
            if deadline is not None
            else absolute_deadline
        )
    from smartsom.telemetry.evaluation_summary import sl_comparison_identity

    comparison_identities = {
        method: sl_comparison_identity(
            evaluation_recipe(a, method), ex, factory=f, workload=w
        )
        for method in methods
    }
    from smartsom.config.evaluation_cases import validate_panel_separation
    from smartsom.config.scenario_snapshot import scenario_from_snapshot

    validate_panel_separation(imported_panels, ex, f, w, actual_drain)
    panel_cases, panel_scenarios, action_seed_is_explicit = [], [], []
    world_indices = []
    for method in methods:
        for index in range(count):
            case = imported[index] if imported else None
            condition = case["condition"] if case else CONDITIONS[index % 4]
            seed = (
                case["seed"]
                if case
                else seed_for(
                    80000 if args.split == ex.validation_split else 90000, index // 4
                )
                % (2**31)
            )
            world = (
                scenario_from_snapshot(case["scenario"])
                if case
                else generate_scenario(
                    *ex.inputs_for(condition, f, w), seed, condition, actual_drain
                )
            )
            # Completion requires a world whose faults cover the protection horizon.
            if (
                case
                and protocol.mode == "completion"
                and world.tick_limit
                != max(d.release_at for d in world.demands) + actual_drain
            ):
                raise ValueError(
                    "completion evaluation requires imported cases frozen through the protection horizon"
                )
            seeds = (
                (
                    case["action_seeds"]
                    if args.selection == "sample"
                    else case["action_seeds"][:1]
                )
                if case
                else range(args.action_seeds)
            )
            for action_seed in seeds:
                world_indices.append(index if imported else index // 4)
                panel_cases.append((method, condition, action_seed, seed))
                panel_scenarios.append(world)
                action_seed_is_explicit.append(case is not None)
    store = case_store(
        root,
        paths,
        "evaluate",
        panel_cases,
        {
            "selection": args.selection,
            "evaluation_protocol": protocol.model_dump(mode="json", by_alias=True),
            "drain_ticks": actual_drain,
            "training_progress": getattr(args, "training_progress", None),
            "checkpoint_sha256": checkpoint_digest,
            "evaluation_runtime_sha256": runtime_digest,
        },
        scenarios=panel_scenarios,
    )
    from contextlib import ExitStack, closing

    from smartsom.experiments.parallel_evaluation import case_results

    settings = {
        "experiment": str(args.experiment),
        "output": str(root),
        "checkpoint": args.checkpoint,
        "checkpoint_digest": checkpoint_digest,
        "selection": args.selection,
        "split": args.split,
        "numerical_threads": getattr(args, "numerical_threads", 1),
        "drain": actual_drain,
        "original_drain": drain,
        "deadline": deadline,
        "source": manifest["source"],
        "inputs": identity(paths),
    }
    results = [store.rows[i]["payload"] for i in sorted(store.rows)]
    tasks = []
    phase = (
        ex.validation_split if args.split == ex.validation_split else ex.test_directory
    )
    replay_mode = ex.replay.for_split(args.split)
    for i, case in enumerate(panel_cases):
        if i in store.rows:
            continue
        record = replay_mode == "all" or (
            replay_mode == "first"
            and not any(p[:2] == case[:2] for p in panel_cases[:i])
        )
        tasks.append(
            {
                "index": i,
                "case": case,
                "scenario": panel_scenarios[i],
                "explicit_seed": action_seed_is_explicit[i],
                "imported": imported is not None,
                "replay_directory": root / "recordings" / f"case-{i:06d}"
                if record
                else None,
            }
        )

    def progress(case_index, row):
        publish_progress(
            root,
            {
                "status": "evaluating",
                "phase": phase,
                "completed_cases": len(store.rows),
                "total_cases": len(store.cases),
                "inflight_case": case_index,
                **row,
                "tracking": dict(args._tracking.status),
            },
            display_root=args._tracking.root,
        )

    from smartsom.experiments.phase_resources import EvaluationLease

    with ExitStack() as resources:
        lease = EvaluationLease(
            ex.concurrency.evaluation_processes,
            baseline_cpus=1 + ex.sampling.processes,
            budget=ex.resources,
            cpus=ex.concurrency.evaluation_cpus,
        )
        resources.callback(lease.close)
        outcomes = resources.enter_context(
            closing(
                case_results(
                    settings,
                    tasks,
                    (ex, f, w, a, paths),
                    workers=min(
                        ex.concurrency.evaluation_processes,
                        ex.concurrency.evaluation_cpus
                        or ex.concurrency.evaluation_processes,
                    ),
                    on_progress=progress,
                    capacity=lease.target if lease.binding else None,
                )
            )
        )
        for case_index, outcome in outcomes:
            method, condition, action_seed, seed = panel_cases[case_index]
            result, learning, sampling_seed, case_limited = outcome
            if (
                result["termination_reason"] == "drain_deadline"
                and protocol.mode == "completion"
            ):
                result["termination_reason"] = "completion_protection_limit"
            result.update(
                method=method, action_seed=action_seed, selection=args.selection
            )
            result["evaluation_identity"] = {
                "checkpoint_sha256": checkpoint_digest if learning else None,
                "learning": learning,
                "training_context": getattr(args, "training_context", None)
                if learning
                else None,
                "training_progress": getattr(args, "training_progress", None)
                if learning
                else None,
                "method": method,
                "selection": args.selection,
                "policy_seed": sampling_seed,
                "source_sha256": manifest["source"]["implementation_sha256"],
                "runtime_sha256": runtime_digest,
                "execution_platform": environment["platform"],
                "algorithm_sha256": manifest["inputs"]["algorithm"],
                "split": args.split,
                "sl_comparison": comparison_identities[method]
                if learning
                else {"status": "unsupported"},
                "world_index": world_indices[case_index],
                "world_sha256": fingerprint(store.cases[case_index]["scenario"]),
                "evaluation_protocol": protocol.model_dump(mode="json", by_alias=True),
            }
            if result.get("wall_clock_fragment"):
                result["wall_limit_scope"] = "case" if case_limited else "panel"
                if not case_limited:
                    write_json(root / "interrupted-case.json", result)
                    raise TimeoutError(
                        "evaluation wall budget reached; partial evidence retained"
                    )
                # The configured case protection is an observed, censored outcome.
                # Receipt it once and attempt later cases; a panel budget interruption
                # above remains uncommitted so continuation can retry that case.
                result["status"] = "truncated"
            commit_case(store, case_index, result)
            args._tracking.log(
                {
                    "phase": phase,
                    "selection": args.selection,
                    "training_progress": getattr(args, "training_progress", None),
                    "case_index": case_index,
                    **result,
                }
            )
            results.append(result)
            publish_cases(root, store)
            publish_progress(
                root,
                {
                    "status": "evaluating",
                    "phase": phase,
                    "completed_cases": len(results),
                    "total_cases": len(store.cases),
                    "cases": len(results),
                    "total": len(store.cases),
                    "tracking": dict(args._tracking.status),
                },
                display_root=args._tracking.root,
            )
    results = [store.rows[i]["payload"] for i in range(len(store.cases))]
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
                "not_drained_rate": 1 - len(complete) / len(rows) if rows else None,
                "mean_tardiness": sum(r["mean_tardiness"] for r in complete) / len(rows)
                if rows and len(complete) == len(rows)
                else None,
                "joint_cost": sum(r["joint_cost"] for r in complete) / len(rows)
                if rows and len(complete) == len(rows)
                else None,
                "passing_rate": sum(r["failed"] for r in rows)
                / sum(r["completed"] for r in rows)
                if sum(r["completed"] for r in rows)
                else None,
            }
            from smartsom.telemetry.evaluation_summary import completion_tail_summary

            summary[method][condition]["completion_tail"] = completion_tail_summary(
                rows
            )
            if summary[method][condition]["passing_rate"] is not None:
                summary[method][condition]["passing_rate"] = (
                    1 - summary[method][condition]["passing_rate"]
                )
            summary[method][condition]["observed_passing_rate"] = summary[method][
                condition
            ]["passing_rate"]
            if len(complete) != len(rows):
                summary[method][condition]["passing_rate"] = None
    from smartsom.telemetry.evaluation_summary import hierarchical, paired_robustness

    research = {
        method: {
            condition: hierarchical(
                [
                    r
                    for r in results
                    if r["method"] == method
                    and (condition == "all" or r["condition"] == condition)
                ]
            )
            for condition in (*CONDITIONS, "all")
        }
        for method in methods
    }
    from smartsom.telemetry.evaluation_summary import (
        reference_definition,
        reference_worlds,
    )

    reference_panel = {
        **reference_definition(ex, f, w, a),
        "worlds": reference_worlds(results),
    }
    if protocol.learning_target is not None and args.split == ex.validation_split:
        reference = json.loads(
            (
                paths["experiment"].parent / protocol.learning_target.reference
            ).read_text()
        )
        expected_worlds = [
            tuple(world) for world in reference["reference_panel"]["worlds"]
        ]
        if reference_panel["worlds"] != expected_worlds:
            raise ValueError(
                "CAR periodic_evaluation frozen worlds differ from the Rule target"
            )
        if manifest["source"]["implementation_sha256"] != reference["source_sha256"]:
            raise ValueError(
                "CAR periodic_evaluation source differs from the Rule target"
            )
    write_json(
        root / "research-summary.json",
        {
            "schema": "smartsom.evaluation-summary/v1",
            "reference_panel": reference_panel,
            "split": args.split,
            "selection": args.selection,
            "source_sha256": manifest["source"]["implementation_sha256"],
            "methods": research,
            "robustness": paired_robustness(results),
            "protocol": protocol.model_dump(mode="json", by_alias=True),
        },
    )
    for method, conditions in research.items():
        for condition, values in conditions.items():
            summary[method][condition].update(values["metrics"])
    args._tracking.log(
        {
            "phase": phase,
            "kind": "summary",
            "training_progress": getattr(args, "training_progress", None),
            "selection": args.selection,
            "methods": research,
        }
    )
    write_json(root / "summary.json", summary)
    publish_progress(
        root,
        {
            "status": "completed",
            "phase": phase,
            "completed_cases": len(results),
            "total_cases": len(store.cases),
            "cases": len(results),
            "tracking": dict(args._tracking.status),
        },
        display_root=args._tracking.root,
    )
