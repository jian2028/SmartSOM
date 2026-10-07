"""Composable v3 lifecycle: immutable inputs, model packages and complete resume."""

import copy
import hashlib
import json
import pickle
import random
import shutil
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from statistics import mean
from uuid import uuid4

from smartsom._filesystem import atomic_replace, native_path
from smartsom.algorithms.production_composition import (
    BoundaryCoordinator,
    replay_boundary,
)
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.experiment_v3 import PreparedComposition
from smartsom.config.production import named_seed, scenario_from_snapshot
from smartsom.domain.production_decisions import ACTION_CONTRACT, OBSERVATION_CONTRACT
from smartsom.domain.travel_time import physical_contract, validate_model_contract
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.control import StopRequested, boundary
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.learning.extensions import dispatcher_pickup_opportunity
from smartsom.learning.production_contract import factory_identity
from smartsom.telemetry.workflow import describe_prepared
from smartsom.trace.performance import (
    fixed_demand_performance,
    tardiness_totals,
    theoretical_reference,
)


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    return digest(
        [
            (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.rglob("*.py"))
        ]
    )


def model_identity(model):
    if model is None:
        return None
    metadata = model["metadata"]
    return {
        "source": model.get("original_source", model["source"]),
        "package_sha256": model.get("package_sha256"),
        **{
            k: metadata.get(k)
            for k in (
                "schema",
                "role",
                "algorithm",
                "backend",
                "weights_sha256",
                "encoder_sha256",
                "source_update",
                "group",
                "scientific_sha256",
                "physical_contract",
                "transport_matrix_sha256",
            )
        },
    }


def bindings(prepared):
    composition = json.loads(prepared.composition_json)
    if composition.get("controller"):
        return {
            r: {"default": "central", "overrides": {}}
            for r in ("machine", "buffer", "dispatcher", "mover")
        }
    return composition["bindings"]


def policies_for(prepared, training=False):
    declarations = json.loads(prepared.policies_json)
    if all(d["implementation"]["kind"] == "rule" for d in declarations.values()):
        return {
            g: RulePolicy(
                d["role"],
                d["implementation"]["name"],
                named_seed(prepared.config.seed, "policy:" + g),
                d["implementation"]["parameters"],
                version=d["implementation"].get("version"),
                code_sha256=d["implementation"].get("code_sha256"),
                frozen_identity=d.get("resolved_rule"),
            )
            for g, d in declarations.items()
        }, {}
    from smartsom.learning.production_inference import build_groups

    return build_groups(prepared, training=training)


def parallel_sampling_issue(policies):
    """Identify components whose mutable state cannot be merged across workers."""
    for group, policy in policies.items():
        if (
            isinstance(policy, RulePolicy)
            and policy.registration is not None
            and policy.registration.stateful
        ):
            return f"parallel sampling cannot merge stateful rule group {group}"
        if hasattr(policy, "encoder") and any(
            component.registration.stateful
            for component in policy.encoder.extensions.components.values()
        ):
            return f"parallel sampling cannot merge stateful observation group {group}"
    return None


def prepared_from_run(root):
    root = Path(root).resolve()
    snapshot = root / "config/prepared.json"
    adaptive = not native_path(snapshot).is_file()
    if adaptive:
        # Tune attempts preserve the scientific input separately from their
        # effective resource allocation. Never treat an arbitrary backup as input.
        snapshot = root / "config/original-prepared.json"
    data = json.loads(native_path(snapshot).read_text(encoding="utf-8"))
    if adaptive:
        from smartsom.experiments.tuning_session import verify_identity

        record = json.loads(
            (native_path(root / "run.json")).read_text(encoding="utf-8")
        )
        if not record.get("tuning", {}).get("experiment_id"):
            raise ValueError("alternate preparation requires a registered Tune attempt")
        pointer = json.loads(
            (native_path(root / "checkpoints/adaptive-recovery.json")).read_text(
                encoding="utf-8"
            )
        )
        checkpoint = (root / "checkpoints" / pointer["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(root / "checkpoints"):
            raise ValueError("adaptive checkpoint escapes its attempt")
        verify_identity(PreparedComposition(**data), record, checkpoint)
    verify_prepared_rules(PreparedComposition(**data))
    declarations = json.loads(data["policies_json"])
    for declaration in declarations.values():
        model = declaration.get("resolved_model")
        if model and model["source"].startswith("$RUN/"):
            model["source"] = str(root / model["source"][5:])
    from smartsom.config.experiment_v3 import model_location
    from smartsom.config.policies import ModelSelector

    for declaration in declarations.values():
        model = declaration.get("resolved_model")
        if model:
            actual = model_location(ModelSelector(source=model["source"]))
            if digest(actual["metadata"]) != digest(model["metadata"]):
                raise ValueError("archived partner model metadata changed")
            if (
                model.get("package_sha256")
                and actual.get("package_sha256") != model["package_sha256"]
            ):
                raise ValueError("archived partner model package changed")
    data["policies_json"] = canonical_json(declarations)
    return PreparedComposition(**data)


def verify_prepared_rules(prepared):
    """Explicitly loaded rule modules are pinned in new authoring snapshots."""
    from smartsom.algorithms.rule_registry import freeze_rule, verify_rule_modules

    recipe = json.loads(prepared.training_inputs_json)
    modules = recipe.get("authoring", {}).get("extension_modules", [])
    verify_rule_modules(modules, load=True)
    for declaration in json.loads(prepared.policies_json).values():
        frozen = declaration.get("resolved_rule")
        if frozen is not None:
            impl = declaration["implementation"]
            current = freeze_rule(
                declaration["role"],
                impl["name"],
                version=impl.get("version"),
                parameters=impl["parameters"],
                code_sha256=frozen["code_sha256"],
            )
            if current != frozen:
                raise ValueError("registered rule identity changed")


def archive_inputs(root, prepared):
    declarations = json.loads(prepared.policies_json)
    for group, declaration in declarations.items():
        model = declaration.get("resolved_model")
        if not model:
            continue
        source = Path(model["source"])
        target = root / "dependencies" / group
        native_path(target.parent).mkdir(exist_ok=True)
        if native_path(source).is_file():
            target = target.with_suffix(".zip")
            shutil.copyfile(native_path(source), native_path(target))
        else:
            native_path(target).mkdir()
            for name in ("model.json", "weights.pt", "encoder.json"):
                shutil.copyfile(native_path(source / name), native_path(target / name))
        model["original_source"] = model["source"]
        model["source"] = "$RUN/" + str(target.relative_to(root))
    saved = replace(prepared, policies_json=canonical_json(declarations))
    write_json(root / "config/prepared.json", asdict(saved))
    (native_path(root / "config/experiment.json")).write_text(
        prepared.config_json + "\n", encoding="utf-8"
    )
    return prepared_from_run(root)


def allocate(prepared, kind):
    config = prepared.config
    root = Path(config.output.root).resolve() / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + config.output.name
        + "-"
        + uuid4().hex[:10]
    )
    native_path(root).mkdir(parents=True)
    from smartsom.telemetry.runtime import bind

    bind(root)
    for folder in (
        "config",
        "checkpoints",
        "evidence",
        "evaluation",
        "logs",
        "reports",
    ):
        (native_path(root / folder)).mkdir(exist_ok=True)
    record = {
        "schema": (
            "smartsom.experiment/v4"
            if config.schema_id == "smartsom.execution-config/v2"
            else "smartsom.experiment/v3"
        ),
        "kind": kind,
        "status": "running",
        "id": root.name,
        "scientific_sha256": prepared.scientific_sha256,
        "source": source_identity(),
        "physical_ticks": 0,
        "updates": 0,
        "provider": "composable",
        "reward_contract": primitive(config.training.reward.task)
        if config.training and config.training.reward.task is not None
        else "v3-blind-output-legacy-coefficients/v1",
        "physical_contract": physical_contract(prepared.scenario),
        "implementation_sha256": implementation_identity(),
        "paths": {"config": "config/prepared.json"},
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(root / "run.json", record)
    try:
        prepared = archive_inputs(root, prepared)
        from smartsom.telemetry.runtime import configure_workflow

        configure_workflow(prepared, kind)
        record["effective_policies"] = {
            g: {
                "role": d["role"],
                "kind": d["implementation"]["kind"],
                "model": model_identity(d.get("resolved_model")),
            }
            for g, d in json.loads(prepared.policies_json).items()
        }
        write_json(root / "run.json", record)
    except BaseException as exc:
        record.update(
            status="failed",
            failure={"exception": type(exc).__name__, "message": str(exc)},
        )
        write_json(root / "run.json", record)
        exc.run_dir = root
        raise
    return root, record, prepared


def package(directory, policy, prepared, group, update, *, partners):
    import torch

    directory = native_path(directory)

    native_path(directory).mkdir(parents=True, exist_ok=True)
    torch.save(policy.network.state_dict(), directory / "weights.pt")
    write_json(directory / "encoder.json", policy.encoder.state_dict())
    metadata = {
        **copy.deepcopy(policy.metadata),
        "schema": "smartsom.component-model/v3",
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "factory_identity": factory_identity(prepared.scenario.factory),
        "physical_contract": physical_contract(prepared.scenario),
        "transport_matrix_sha256": digest(
            primitive(prepared.scenario.transport_matrix)
        ),
        "weights_file": "weights.pt",
        "weights_sha256": hashlib.sha256(
            (native_path(directory / "weights.pt")).read_bytes()
        ).hexdigest(),
        "encoder_sha256": hashlib.sha256(
            (native_path(directory / "encoder.json")).read_bytes()
        ).hexdigest(),
        "context_size": policy.encoder.context_size,
        "group": group,
        "source_update": update,
        "partners": partners,
        "scientific_sha256": prepared.scientific_sha256,
        "task_reward_contract": primitive(prepared.config.training.reward.task)
        if prepared.config.training
        else None,
    }
    if policy.metadata["role"] == "central":
        metadata["schema"] = "smartsom.central-model/v3"
    write_json(directory / "model.json", metadata)


def checkpoint_path(root, selection):
    root = Path(root)
    if selection in ("best", "last"):
        path = root / "checkpoints" / (selection + ".json")
        if not native_path(path).is_file():
            raise ValueError(f"{selection} checkpoint does not exist")
        selection = json.loads(native_path(path).read_text(encoding="utf-8"))[
            "checkpoint"
        ]
    if not selection.startswith("update-") or not selection[7:].isdigit():
        raise ValueError("invalid checkpoint identifier")
    path = root / "checkpoints" / selection
    if not (native_path(path / "snapshot.json")).is_file():
        raise ValueError("incomplete experiment snapshot")
    return path


def evaluation_recipe(prepared, checkpoint):
    declarations = json.loads(prepared.policies_json)
    for group, declaration in declarations.items():
        if declaration["implementation"]["kind"] == "rule":
            continue
        location = checkpoint / (
            "controllers/central"
            if json.loads(prepared.composition_json).get("controller")
            else "groups/" + group
        )
        from smartsom.config.experiment_v3 import model_location
        from smartsom.config.policies import ModelSelector

        resolved = model_location(ModelSelector(source=str(location)))
        validate_model_contract(resolved["metadata"], prepared.scenario)
        declaration["implementation"] = {
            "kind": "model",
            "model": {"source": str(location)},
        }
        declaration["resolved_model"] = resolved
    return replace(prepared, policies_json=canonical_json(declarations))


def _worker_tick(sim, policies, routes, matching, episode):
    if isinstance(policies, bytes):
        policies = pickle.loads(policies)
    coordinator = BoundaryCoordinator(sim, policies, routes, matching, episode=episode)
    result = coordinator.tick()
    return (
        sim,
        result,
        coordinator.records,
        pickle.dumps(coordinator.state_dict(), protocol=5),
    )


def episode_metrics(
    sim, *, case="0", replication=0, seed=None, error=None, task_reward=None
):
    delivered = len(sim.completed)
    # Failed Output attempts remain in jobs after their replacement succeeds.
    # Completion statistics count the qualified attempt once per demand.
    completions = {}
    for job in sim.jobs.values():
        demand = job["demand"]
        if (
            demand in sim.completed
            and sim.roles.get(job["location"]) == "system_output"
            and job["quality"] == "PASS"
        ):
            completions[demand] = min(
                completions.get(demand, job["since"]), job["since"]
            )
    flow = [
        tick - sim.demands[demand].release_at for demand, tick in completions.items()
    ]
    waiting = sum(
        sim.metrics.get(k, 0) for k in ("reservation_wait_ticks", "destination_waiting")
    )
    blind = getattr(sim, "protocol", None) is not None
    if blind:
        completions = dict(sim.shipment_times)
        flow = [tick - sim.demands[d].release_at for d, tick in completions.items()]
    quality = (
        sim.privileged_output_quality()
        if blind
        else {
            "shipped": sim.metrics.get("submitted", 0),
            "good_shipped": delivered,
            "bad_shipped": sim.metrics.get("output_rejected", 0),
            "passing_rate": delivered / sim.metrics["submitted"]
            if sim.metrics.get("submitted")
            else None,
        }
    )
    delivered_due = [
        (tick, sim.demands[demand].due_at) for demand, tick in completions.items()
    ]
    task_values = {}
    if task_reward is not None:
        from smartsom.experiments.task_reward import shipment_reward

        task_values = {
            "return": shipment_reward(task_reward, sim.privileged_task_totals()),
            "task_reward_contract": primitive(task_reward),
            "task_window_complete": sim.done,
            "accumulated_overdue_time": sim._overdue_time,
            "raw_legacy_return": sim.total_reward,
        }
    return {
        "case_id": case,
        "algorithm_id": "composition",
        "replication": replication,
        "seed": seed,
        "status": "exception" if error else sim.status,
        "engineering_failure": bool(error),
        "exception": error,
        "return": sim.total_reward,
        **task_values,
        "delivered": delivered,
        "flow_time": mean(flow) if flow else None,
        "waiting": waiting,
        "makespan": (
            max(completions.values(), default=0)
            if blind and len(completions) == len(sim.demands)
            else None
            if blind
            else sim.tick
            if sim.status == "completed"
            else None
        ),
        "fulfillment_complete": len(completions) == len(sim.demands),
        "throughput": delivered / sim.tick if sim.tick else None,
        **tardiness_totals(delivered_due),
        **fixed_demand_performance(
            {d: demand.due_at for d, demand in sim.demands.items()},
            completions,
            sim.tick,
        ),
        "passing_rate": quality["passing_rate"],
        "privileged_output_quality": quality,
        "output_submitted": sim.metrics.get("submitted", 0),
        "output_qualified": quality["good_shipped"],
        **(
            {"output_bad_shipped": quality["bad_shipped"]}
            if blind
            else {"output_rejected": sim.metrics.get("output_rejected", 0)}
        ),
        "pre_output_scrap": sim.metrics.get("pre_output_scrap", 0),
        "theoretical": theoretical_reference(sim.scenario),
        "physical_ticks": sim.tick,
        "truncated": sim.status == "truncated",
        "metrics": dict(sim.metrics),
        **(
            {
                "machine_utilization": {
                    m: sim.metrics[f"machine_processing_ticks:{m}"] / max(1, sim.tick)
                    for m in sim.machines
                },
                "mean_wip": sim.metrics["wip_ticks"] / max(1, sim.tick),
                "job_waiting_ticks": sim.metrics["job_waiting_ticks"],
                "transport_waiting_ticks": sim.metrics["matrix_port_wait_ticks"],
                "travel_ticks": sim.metrics["matrix_travel_ticks"],
            }
            if sim.scenario.transport_matrix
            else {}
        ),
    }


def evaluate_cases(
    prepared,
    cases,
    *,
    directory=None,
    on_progress=None,
    validation=False,
    label=None,
    controls=None,
):
    from smartsom.telemetry.runtime import emit

    if not validation and prepared.config.evaluation.render_mode and controls is None:
        import threading
        from contextvars import copy_context

        from smartsom.experiments.production import RunControls
        from smartsom.studio.playback import live_window

        if threading.current_thread() is not threading.main_thread():
            raise ValueError("human rendering must be launched from the main thread")
        controls = RunControls()
        results, errors = [], []

        def worker():
            try:
                results.append(
                    evaluate_cases(
                        prepared,
                        cases,
                        directory=directory,
                        on_progress=on_progress,
                        validation=validation,
                        label=label,
                        controls=controls,
                    )
                )
            except BaseException as exc:
                controls.error = exc
                errors.append(exc)
            finally:
                controls.finished = True

        context = copy_context()
        thread = threading.Thread(
            target=lambda: context.run(worker), name="smartsom-v3-evaluation"
        )
        live_window(prepared.scenario.factory, controls, thread)
        thread.join()
        if errors:
            raise errors[0]
        return results[0]

    rows = []
    phase = "validation" if validation else "evaluation"
    workflow = describe_prepared(prepared, "training" if validation else "evaluation")
    last_progress = 0.0

    def report(sim, *, force=False, ended=False):
        nonlocal last_progress
        now = time.monotonic()
        if force or now - last_progress >= 2:
            emit(
                "training" if validation else "evaluation",
                {
                    "stage": phase,
                    "evaluation_kind": label or phase,
                    **(
                        {}
                        if validation
                        else {
                            "evaluation_completed": sum(
                                r["status"] == "completed" for r in rows
                            )
                        }
                    ),
                    phase + "_finished": len(rows),
                    phase + "_requested": len(cases),
                    phase + "_tick": sim.tick,
                    phase + "_tick_limit": sim.scenario.tick_limit,
                    phase + "_case_active": not ended,
                    "workflow": workflow,
                },
                **(
                    {}
                    if validation
                    else {"total": len(cases), "unit": "evaluation episodes"}
                ),
            )
            last_progress = now

    matching = json.loads(prepared.composition_json)["matching"]["name"]
    for index, case in enumerate(cases):
        if not validation:
            boundary(directory.parent if directory else None)
        policies, _ = policies_for(prepared, training=False)
        sim = ProductionSimulator(
            scenario_from_snapshot(case["scenario"]), contract="v3"
        )
        options = prepared.config.evaluation
        show = (
            controls is not None
            and case["case"] == (options.render_case or cases[0]["case"])
            and case["replication"] + 1 == options.render_replication
        )
        if show:
            controls.context = {
                "case": case["case"],
                "seed": case["seed"],
                "replication": case["replication"] + 1,
            }
            controls.latest = {"tick": 0, "state": sim.snapshot(), "events": []}
        coordinator = BoundaryCoordinator(sim, policies, bindings(prepared), matching)
        before = {
            g: p.fingerprint() for g, p in policies.items() if hasattr(p, "fingerprint")
        }
        from smartsom.trace.production import Recorder

        recorder = None
        if directory is not None:
            target = directory / f"case-{index:04d}"
            recorder = Recorder(
                target,
                {
                    "scenario": primitive(sim.scenario),
                    "algorithm": {"provider": "composable"},
                    "composition": json.loads(prepared.composition_json),
                },
                sim.snapshot(),
                record=prepared.config.evaluation.record,
                source=source_identity(),
            )
            recorder.manifest["action_contract"] = ACTION_CONTRACT
        replay = (
            ProductionSimulator(sim.scenario, contract="v3")
            if (
                prepared.config.validation if validation else prepared.config.evaluation
            ).full_replay
            else None
        )
        report(sim, force=True)
        dispatch_diagnostics = {
            "eligible_pickup_boundaries": 0,
            "eligible_empty_agv_decisions": 0,
            "missed_pickup_boundaries": 0,
            "first_reservation_tick": None,
            "first_pickup_tick": None,
        }
        try:
            while not sim.done:
                if not validation:
                    boundary(directory.parent if directory else None)
                if show and not controls.permission():
                    raise StopRequested("stopped from live window")
                public_before = sim.protocol.public_view()
                record = coordinator.tick()
                eligible, missed = dispatcher_pickup_opportunity(
                    public_before, coordinator.records
                )
                dispatch_diagnostics["eligible_empty_agv_decisions"] += eligible
                dispatch_diagnostics["eligible_pickup_boundaries"] += int(eligible > 0)
                dispatch_diagnostics["missed_pickup_boundaries"] += int(missed)
                for event in record.get("events", ()):
                    if (
                        event["kind"] == "source_reserved"
                        and dispatch_diagnostics["first_reservation_tick"] is None
                    ):
                        dispatch_diagnostics["first_reservation_tick"] = event["tick"]
                    if (
                        event["kind"] == "pickup_started"
                        and dispatch_diagnostics["first_pickup_tick"] is None
                    ):
                        dispatch_diagnostics["first_pickup_tick"] = event["tick"]
                if show:
                    controls.latest = {
                        "tick": sim.tick,
                        "state": sim.snapshot(),
                        "events": record.get("events", []),
                    }
                    if controls.delay:
                        time.sleep(controls.delay)
                report(sim)
                if recorder:
                    recorder.append(record)
                if replay:
                    replay_boundary(replay, record)
            row = episode_metrics(
                sim,
                case=case["case"],
                replication=case["replication"],
                seed=case["seed"],
                task_reward=prepared.config.training.reward.task
                if prepared.config.training
                else None,
            )
            row["dispatcher_diagnostics"] = dispatch_diagnostics
            if any(before[g] != policies[g].fingerprint() for g in before):
                raise ValueError("evaluation changed frozen model/normalization")
        except StopRequested:
            if recorder:
                recorder.finish("interrupted", "stop requested")
            raise
        except Exception as exc:
            row = episode_metrics(
                sim,
                case=case["case"],
                replication=case["replication"],
                seed=case["seed"],
                error=f"{type(exc).__name__}: {exc}",
                task_reward=prepared.config.training.reward.task
                if prepared.config.training
                else None,
            )
            row["dispatcher_diagnostics"] = dispatch_diagnostics
        rows.append(row)
        report(sim, force=True, ended=True)
        if directory is not None:
            recorder.finish(
                sim.status if not row["engineering_failure"] else "failed",
                row["exception"],
            )
            write_json(target / "result.json", row)
            row["run_dir"] = str(target.relative_to(directory.parent))
        if on_progress:
            on_progress({"evaluation_finished": len(rows), "requested": len(cases)})
    return rows


def summarize(rows):
    return {
        "requested": len(rows),
        "completed": sum(r["status"] == "completed" for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "exceptions": sum(r["engineering_failure"] for r in rows),
        "mean_return": mean(r["return"] for r in rows) if rows else None,
        "mean_delivery": mean(r["delivered"] for r in rows) if rows else None,
        "mean_flow_time": mean(
            r["flow_time"] for r in rows if r["flow_time"] is not None
        )
        if any(r["flow_time"] is not None for r in rows)
        else None,
        "mean_waiting": mean(r["waiting"] for r in rows) if rows else None,
        "mean_makespan": mean(r["makespan"] for r in rows if r["makespan"] is not None)
        if any(r["makespan"] is not None for r in rows)
        else None,
        **{
            f"mean_{key}": mean(values) if values else None
            for key, values in (
                (
                    key,
                    [r[key] for r in rows if r.get(key) is not None],
                )
                for key in (
                    "throughput",
                    "total_tardiness",
                    "tardy_jobs",
                    "passing_rate",
                    "on_time_delivery_fraction",
                    "total_tardiness_lower_bound",
                    "delivered_total_tardiness",
                )
            )
        },
        # Whole-suite fixed-job means require every case to finish; legacy
        # mean_makespan above remains explicitly a completed-case statistic.
        **{
            f"mean_{key}": mean(r[key] for r in rows)
            if rows
            and all(
                r.get(key) is not None and not r["engineering_failure"] for r in rows
            )
            else None
            for key in ("fixed_job_makespan", "fixed_job_total_tardiness")
        },
        "fixed_job_completed_cases": sum(
            bool(r.get("makespan_complete")) and not r["engineering_failure"]
            for r in rows
        ),
        "passing_rate_defined_cases": sum(
            r.get("passing_rate") is not None for r in rows
        ),
        # A declared bound for these cases, never a target; it carries the
        # assumptions it drops.
        "theoretical": next(
            (r["theoretical"] for r in rows if r.get("theoretical")), None
        ),
    }


def prepare_evaluation(
    prepared=None,
    *,
    source=None,
    selection="last",
    output_root=None,
    options=None,
):
    checkpoint = None
    if source is not None:
        prepared = prepared_from_run(source)
        checkpoint = checkpoint_path(source, selection)
        prepared = evaluation_recipe(prepared, checkpoint)
        if options is not None:
            from smartsom.config.production import (
                ScenarioFile,
                WorkloadFile,
                materialize,
            )

            if options.baselines:
                raise ValueError(
                    "v3 comparison partners are selected with evaluate --config"
                )
            originals = json.loads(prepared.evaluation_json)
            by_case = {}
            for case in originals:
                by_case.setdefault(case["case"], case)
            cases = []
            if prepared.config.schema_id in {
                "smartsom.execution-config/v1",
                "smartsom.execution-config/v2",
            }:
                frozen = prepared.config.evaluation
                if (
                    options.scenarios
                    or options.seed != frozen.seed
                    or options.replications > frozen.replications
                ):
                    raise ValueError(
                        "author source evaluation uses frozen data; create a new Experiment to change cases, data seed or add replications"
                    )
                cases = [
                    case
                    for case in originals
                    if case["replication"] < options.replications
                ]
            elif options.scenarios:
                from smartsom.config.experiment_v3 import world

                for index, scenario_path in enumerate(options.scenarios):
                    for replication in range(options.replications):
                        seed = named_seed(
                            options.seed, f"evaluation:{index}:{replication}"
                        )
                        cases.append(
                            {
                                "case": str(index),
                                "replication": replication,
                                "seed": seed,
                                "scenario": primitive(world(scenario_path, seed)),
                            }
                        )
            else:
                for index, case in by_case.items():
                    recipe = case["recipe"]
                    factory = scenario_from_snapshot(case["scenario"]).factory
                    for replication in range(options.replications):
                        seed = named_seed(
                            options.seed, f"evaluation:{index}:{replication}"
                        )
                        scenario = materialize(
                            factory,
                            WorkloadFile.model_validate_json(
                                canonical_json(recipe["workload"])
                            ),
                            ScenarioFile.model_validate_json(
                                canonical_json(recipe["settings"])
                            ),
                            seed,
                        )
                        cases.append(
                            {
                                "case": index,
                                "replication": replication,
                                "seed": seed,
                                "scenario": primitive(scenario),
                                "recipe": recipe,
                            }
                        )
            config = json.loads(prepared.config_json)
            config["evaluation"] = primitive(options)
            prepared = replace(
                prepared,
                config_json=canonical_json(config),
                evaluation_json=canonical_json(cases),
            )
    if prepared is None:
        raise ValueError("evaluate needs a composition or experiment snapshot")
    for declaration in json.loads(prepared.policies_json).values():
        if declaration["implementation"]["kind"] == "model":
            validate_model_contract(
                declaration["resolved_model"]["metadata"], prepared.scenario
            )
    if not json.loads(prepared.evaluation_json):
        raise ValueError(
            "no frozen evaluation cases; this training run cannot be evaluated "
            "from its saved snapshot"
        )
    if output_root is not None:
        config = json.loads(prepared.config_json)
        config["output"]["root"] = str(output_root)
        prepared = replace(prepared, config_json=canonical_json(config))
    if any(
        physical_contract(scenario_from_snapshot(case["scenario"]))
        != physical_contract(prepared.scenario)
        for case in json.loads(prepared.evaluation_json)
    ):
        raise ValueError("evaluation transport/processing contract is incompatible")
    origin = prepared.scientific_sha256
    for case in json.loads(prepared.evaluation_json):
        if factory_identity(
            scenario_from_snapshot(case["scenario"]).factory
        ) != factory_identity(prepared.scenario.factory):
            raise ValueError("evaluation factory structure is incompatible")
    prepared = replace(
        prepared,
        scientific_sha256=digest(
            {
                "config": json.loads(prepared.config_json),
                "composition": json.loads(prepared.composition_json),
                "policies": json.loads(prepared.policies_json),
                "inputs": json.loads(prepared.evaluation_json),
            }
        ),
    )
    return prepared, checkpoint, origin


def evaluate(
    prepared=None,
    *,
    source=None,
    selection="last",
    output_root=None,
    on_progress=None,
    options=None,
    purpose=None,
):
    from smartsom.experiments.evaluation import EvaluationResult

    prepared, checkpoint, origin = prepare_evaluation(
        prepared,
        source=source,
        selection=selection,
        output_root=output_root,
        options=options,
    )
    root, record, prepared = allocate(prepared, "evaluation")
    if purpose is not None:
        record["purpose"] = purpose
        write_json(root / "run.json", record)
    from smartsom.config.travel_time import matrix_summary

    record["transport_inputs"] = {
        "source": matrix_summary(prepared.scenario.transport_matrix),
        "cases": [
            matrix_summary(scenario_from_snapshot(case["scenario"]).transport_matrix)
            for case in json.loads(prepared.evaluation_json)
        ],
    }
    record["source_scientific_sha256"] = origin
    record["evaluation_inputs_sha256"] = digest(json.loads(prepared.evaluation_json))
    try:
        rows = evaluate_cases(
            prepared,
            json.loads(prepared.evaluation_json),
            directory=root / "evidence",
            on_progress=on_progress,
        )
    except BaseException as exc:
        record["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        write_json(root / "run.json", record)
        exc.run_dir = root
        raise
    summary = summarize(rows)
    record.update(
        status="completed" if not summary["exceptions"] else "failed",
        results=rows,
        checkpoint=str(checkpoint) if checkpoint else None,
        composition=json.loads(prepared.composition_json),
        summary=summary,
    )
    write_json(root / "run.json", record)
    write_json(root / "summary.json", summary)
    return EvaluationResult(
        root,
        record["status"],
        summary["completed"],
        len(rows) - summary["completed"],
        summary["exceptions"],
        tuple(rows),
        checkpoint,
    )


def _sampling_threads(threads, rule_modules=(), observation_identity=None):
    """Preserve the frozen child-thread allocation across driver resizes."""
    import os

    from smartsom.algorithms.rule_registry import verify_rule_modules

    verify_rule_modules(rule_modules, load=True)

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    if observation_identity is not None:
        from smartsom.learning.physical_job_observation import SCHEMA, install

        choices = {
            SCHEMA + "/inspection=True": True,
            SCHEMA + "/inspection=False": False,
        }
        if observation_identity not in choices:
            raise ValueError("unknown sampling observation identity")
        install(include_inspection=choices[observation_identity])

    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    # Keep the controller alive for the worker lifetime.
    global _SAMPLING_LIMITS
    _SAMPLING_LIMITS = threadpool_limits(limits=threads)


class TrainingSession:
    def __init__(self, prepared, root, record, *, sampling_numerical_threads=None):
        import torch

        from smartsom.learning.production_collection import PhysicalCollector, Replay

        self.prepared, self.root, self.record = prepared, root, record
        self.display_workflow = describe_prepared(prepared, "training")
        self.config, self.settings = prepared.config, prepared.config.training
        self.parameters = json.loads(prepared.parameters_json)
        torch.set_num_threads(self.config.runtime.numerical_threads)
        self.policies, self.learners = policies_for(prepared, training=True)
        self.central = bool(json.loads(prepared.composition_json).get("controller"))
        self.collector = PhysicalCollector(
            self.settings.groups, self.settings.gamma, central=self.central
        )
        self.routes = bindings(prepared)
        self.matching = json.loads(prepared.composition_json)["matching"]["name"]
        self.parallel_sampling = bool(
            self.config.runtime.sampling_processes and self.config.runtime.num_envs > 1
        )
        if self.parallel_sampling:
            issue = parallel_sampling_issue(self.policies)
            if issue:
                raise ValueError(issue)
        self.sampler_states = []
        if self.parallel_sampling:
            for index in range(self.config.runtime.num_envs):
                partners = copy.deepcopy(self.policies)
                for group, policy in partners.items():
                    seed = named_seed(self.config.seed, f"sampler:{index}:{group}")
                    if isinstance(policy, RulePolicy):
                        policy.seed = seed
                        policy.reset()
                    else:
                        policy.generator.manual_seed(seed % (2**63 - 1))
                        policy.random.seed(seed)
                self.sampler_states.append(
                    {group: policy.state_dict() for group, policy in partners.items()}
                )
        self.episodes = [0] * self.config.runtime.num_envs
        self.sims = [self.new_sim(i) for i in range(self.config.runtime.num_envs)]
        self.ticks, self.updates, self.cursor = 0, 0, 0
        self.random = random.Random(named_seed(self.config.seed, "optimizer"))
        self.replays = (
            {
                g: Replay(
                    self.parameters["replay_capacity"],
                    named_seed(self.config.seed, "replay:" + g),
                )
                for g in self.settings.groups
            }
            if self.settings.algorithm == "dqn"
            else {}
        )
        self.target_clock = {g: 0 for g in self.settings.groups}
        self.optimizations = {g: 0 for g in self.settings.groups}
        self.initial = {
            g: p.fingerprint()
            for g, p in self.policies.items()
            if hasattr(p, "fingerprint")
        }
        self.frozen = {
            g: self.initial[g] for g in self.initial if g not in self.settings.groups
        }
        self.best_score, self.best_update, self.no_improvement = None, None, 0
        self.history, self.actions, self.episode_results = [], [], []
        self.executor = None
        if self.config.runtime.sampling_processes:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            identities = {
                getattr(getattr(policy, "encoder", None), "observation_identity", None)
                for policy in self.policies.values()
                if hasattr(policy, "encoder")
            }
            if len(identities) > 1:
                raise ValueError(
                    "sampling policies have different observation identities"
                )
            observation_identity = next(iter(identities), None)
            self.executor = ProcessPoolExecutor(
                self.config.runtime.sampling_processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_sampling_threads,
                initargs=(
                    sampling_numerical_threads or self.config.runtime.numerical_threads,
                    json.loads(prepared.training_inputs_json)
                    .get("authoring", {})
                    .get("extension_modules", ()),
                    observation_identity,
                ),
            )
        from smartsom.config.extensions import ExtensionSpec
        from smartsom.learning.extensions import ExtensionsRuntime

        provider = (
            "sb3.maskable_ppo"
            if self.settings.backend == "sb3"
            else "rllib.ppo"
            if self.central
            else "rllib.resource_" + self.settings.algorithm
        )
        self.reward_runtime = ExtensionsRuntime(
            ExtensionSpec(reward=self.settings.reward), provider, {}
        )
        self.reward_runtime.begin_episode()

    def new_sim(self, index):
        seed = named_seed(self.config.seed, f"training:{index}:{self.episodes[index]}")
        from smartsom.config.production import ScenarioFile, WorkloadFile, materialize

        recipe = json.loads(self.prepared.training_inputs_json)
        if "data_seed" in recipe:
            seed = named_seed(
                recipe["data_seed"], f"environment:{index}:{self.episodes[index]}"
            )
        scenario = materialize(
            self.prepared.scenario.factory,
            WorkloadFile.model_validate_json(canonical_json(recipe["workload"])),
            ScenarioFile.model_validate_json(canonical_json(recipe["settings"])),
            seed,
        )
        scenario = replace(
            scenario, tick_limit=min(self.settings.max_ticks, scenario.tick_limit)
        )
        return ProductionSimulator(scenario, contract="v3")

    def advance(self):
        index = self.cursor % len(self.sims)
        self.cursor += 1
        sim = self.sims[index]
        if sim.done:
            self.episode_results.append(
                episode_metrics(
                    sim,
                    case=str(index),
                    replication=self.episodes[index],
                    task_reward=self.settings.reward.task,
                )
            )
            self.episodes[index] += 1
            self.sims[index] = sim = self.new_sim(index)
        if self.settings.algorithm == "dqn":
            p = self.parameters
            epsilon = p["epsilon_start"] + min(1, self.ticks / p["epsilon_ticks"]) * (
                p["epsilon_end"] - p["epsilon_start"]
            )
            for group in self.settings.groups:
                self.policies[group].epsilon = epsilon
        coordinator = BoundaryCoordinator(
            sim, self.policies, self.routes, self.matching, episode=self.episodes[index]
        )
        before = sim.protocol.public_view()
        if self.executor:
            sim, outcome, records, state = self.executor.submit(
                _worker_tick,
                sim,
                pickle.dumps(self.policies, protocol=5),
                self.routes,
                self.matching,
                self.episodes[index],
            ).result()
            self.sims[index] = coordinator.sim = sim
            coordinator.records = records
            coordinator.load_state_dict(pickle.loads(state))
        else:
            outcome = coordinator.tick()
        self._finish_tick(index, sim, outcome, coordinator, before)

    def _finish_tick(self, index, sim, outcome, coordinator, before, *, optimize=True):
        from smartsom.learning.extensions import RewardTransition

        after = sim.protocol.public_view()
        task = self.settings.reward.task
        base_reward = outcome["reward"]
        if task is not None:
            from smartsom.experiments.task_reward import shipment_reward

            base_reward = shipment_reward(task, sim.privileged_reward_components())
        transition = RewardTransition(
            before,
            after,
            tuple(coordinator.records),
            base_reward,
            sim.tick,
            sim.status if sim.done else None,
        )
        role_names = (
            [
                self.policies[g].metadata["role"] + "_policy"
                for g in self.settings.groups
            ]
            if not self.central
            else []
        )
        values = self.reward_runtime.rewards(transition, role_names)
        role_rewards = dict(values.roles)
        rewards = {
            g: values.team.learner
            if self.central
            else role_rewards[self.policies[g].metadata["role"] + "_policy"].learner
            for g in self.settings.groups
        }
        bootstrap_inputs = {}
        if (
            task is None
            and sim.status == "truncated"
            and self.settings.algorithm == "dqn"
        ):
            probe = copy.deepcopy(sim)
            probe.scenario = replace(probe.scenario, tick_limit=probe.tick + 1)
            partners = copy.deepcopy(self.policies)
            for policy in partners.values():
                if hasattr(policy, "network"):
                    policy.training = False
                    policy.encoder.set_training(False)
                    policy.network.eval()
            staging = BoundaryCoordinator(
                probe,
                partners,
                self.routes,
                self.matching,
                episode=self.episodes[index],
            )
            staging.tick(stage_only=True)
            for request in staging.records:
                bootstrap_inputs.setdefault(
                    (request["owner"], request["group"]), request["model_input"]
                )
        self.collector.collect(
            index,
            self.episodes[index],
            coordinator,
            outcome,
            self.policies,
            rewards,
            algorithm=self.settings.algorithm,
            terminated=sim.done and (task is not None or sim.status == "completed"),
            truncated=sim.done and task is None and sim.status == "truncated",
            before=before,
            bootstrap_inputs=bootstrap_inputs,
        )
        self.ticks += 1
        self.actions.append(
            {
                "env": index,
                "episode": self.episodes[index],
                "tick": sim.tick,
                "actions": primitive(outcome["actions"]),
                "state_sha256": digest(outcome["state"]),
                "reward": base_reward,
                "reward_contract": primitive(task)
                if task is not None
                else "v3-blind-output-legacy-coefficients/v1",
            }
        )
        if optimize and self.settings.algorithm == "dqn":
            self.optimize_dqn(self.collector.drain("dqn"))

    def advance_wave(self, remaining):
        """Advance distinct environments with frozen weights, then merge by ID."""
        if not self.parallel_sampling:
            self.advance()
            return
        count = min(remaining, len(self.sims))
        if self.settings.algorithm == "dqn":
            # Do not jump over a physical-tick optimizer boundary in a wave.
            interval = self.parameters["train_every_ticks"]
            count = min(count, interval - self.ticks % interval)
            target_interval = self.parameters["target_update_ticks"]
            for clock in self.target_clock.values():
                count = min(count, max(1, clock + target_interval - self.ticks))
        pending = []
        for offset in range(count):
            index = (self.cursor + offset) % len(self.sims)
            sim = self.sims[index]
            if sim.done:
                self.episode_results.append(
                    episode_metrics(
                        sim,
                        case=str(index),
                        replication=self.episodes[index],
                        task_reward=self.settings.reward.task,
                    )
                )
                self.episodes[index] += 1
                self.sims[index] = sim = self.new_sim(index)
            partners = copy.deepcopy(self.policies)
            for group, policy in partners.items():
                policy.load_state_dict(self.sampler_states[index][group])
                if group in self.settings.groups and self.settings.algorithm == "dqn":
                    p = self.parameters
                    policy.epsilon = p["epsilon_start"] + min(
                        1, self.ticks / p["epsilon_ticks"]
                    ) * (p["epsilon_end"] - p["epsilon_start"])
            before = sim.protocol.public_view()
            future = self.executor.submit(
                _worker_tick,
                sim,
                pickle.dumps(partners, protocol=5),
                self.routes,
                self.matching,
                self.episodes[index],
            )
            pending.append((index, before, future))
        self.cursor += count
        # Submission can finish in any order; collection and learner updates cannot.
        for index, before, future in pending:
            sim, outcome, records, state = future.result()
            self.sims[index] = sim
            self.sampler_states[index] = pickle.loads(state)
            coordinator = BoundaryCoordinator(
                sim,
                self.policies,
                self.routes,
                self.matching,
                episode=self.episodes[index],
            )
            coordinator.records = records
            coordinator.load_state_dict(self.sampler_states[index])
            self._finish_tick(index, sim, outcome, coordinator, before, optimize=False)
        if self.settings.algorithm == "dqn":
            self.optimize_dqn(self.collector.drain("dqn"))

    def optimize_dqn(self, groups):
        from smartsom.learning.production_rllib_v3 import replay_batch

        p = self.parameters
        for group, rows in groups.items():
            replay = self.replays[group]
            for row in rows:
                replay.add(row)
            learner = self.learners[group]
            if (
                self.ticks >= p["warmup_ticks"]
                and self.ticks % p["train_every_ticks"] == 0
                and len(replay.rows) >= p["batch_size"]
            ):
                for _ in range(p["gradient_steps"]):
                    learner.update(
                        batch=replay_batch(
                            replay.sample(p["batch_size"]), self.settings.gamma
                        )
                    )
                    self.optimizations[group] += 1
            if self.ticks - self.target_clock[group] >= p["target_update_ticks"]:
                learner.module["default_policy"].target.load_state_dict(
                    self.policies[group].network.state_dict()
                )
                self.target_clock[group] = self.ticks

    def optimize_ppo(self):
        groups = self.collector.drain("ppo", self.parameters["gae_lambda"])
        for group, rows in groups.items():
            if not rows:
                continue
            learner = self.learners[group]
            if self.settings.backend == "sb3":
                before = learner._n_updates
                learner.train_packets(rows, self.parameters, self.random)
                self.optimizations[group] += learner._n_updates - before
            else:
                from smartsom.learning.production_rllib_v3 import packet_batch

                for _ in range(self.parameters["n_epochs"]):
                    order = list(range(len(rows)))
                    self.random.shuffle(order)
                    for start in range(0, len(order), self.parameters["batch_size"]):
                        selected = [
                            rows[i]
                            for i in order[
                                start : start + self.parameters["batch_size"]
                            ]
                        ]
                        learner.update(batch=packet_batch(selected))
                        self.optimizations[group] += 1

    def learner_diagnostics(self):
        from smartsom.learning.production_diagnostics import SCHEMA, summarize

        groups = {}
        for group, learner in self.learners.items():
            row = summarize(getattr(learner, "training_diagnostics", None))
            # Unobserved metrics are unavailable, including zero-update groups.
            names = (
                ("td_loss",)
                if self.settings.algorithm == "dqn"
                else (
                    "actor_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl_k3",
                    "total_loss",
                )
            )
            for name in names:
                row["metrics"].setdefault(
                    name, {"mean": None, "weight": 0, "unavailable_minibatches": 0}
                )
            row.update(
                collector=copy.deepcopy(self.collector.counts.get(group, {})),
                optimization_steps=self.optimizations[group],
                replay_length=len(self.replays[group].rows)
                if group in self.replays
                else None,
                replay_capacity=self.parameters["replay_capacity"]
                if group in self.replays
                else None,
                batch_size=self.parameters["batch_size"],
                target_clock=self.target_clock[group]
                if group in self.replays
                else None,
            )
            groups[group] = row
        return {
            "schema": SCHEMA,
            "aggregation": "cumulative_weighted_minibatches",
            "groups": groups,
        }

    def state_dict(self):
        import numpy as np
        import torch

        learner_states = {}
        for group, learner in self.learners.items():
            if self.settings.backend == "rllib":
                learner_states[group] = learner.get_state()
            else:
                learner_states[group] = {
                    "policy": learner.policy.state_dict(),
                    "optimizer": learner.policy.optimizer.state_dict(),
                    "updates": learner._n_updates,
                }
        return {
            "schema": "smartsom.continuation/v3",
            "action_contract": ACTION_CONTRACT,
            "observation_contract": OBSERVATION_CONTRACT,
            "physical_contract": physical_contract(self.prepared.scenario),
            "task_reward_contract": primitive(self.settings.reward.task),
            "scientific_sha256": self.prepared.scientific_sha256,
            "sims": self.sims,
            "episodes": self.episodes,
            "cursor": self.cursor,
            "ticks": self.ticks,
            "updates": self.updates,
            "policies": {g: p.state_dict() for g, p in self.policies.items()},
            "sampling_layout": {
                "num_envs": self.config.runtime.num_envs,
                "sampling_processes": self.config.runtime.sampling_processes,
            },
            "sampler_states": self.sampler_states,
            "learners": learner_states,
            "collector": self.collector.state_dict(),
            "replays": {g: r.state_dict() for g, r in self.replays.items()},
            "target_clock": self.target_clock,
            "optimizations": self.optimizations,
            "reward": self.reward_runtime.state_dict(),
            "learner_diagnostics": {
                g: copy.deepcopy(getattr(learner, "training_diagnostics", None))
                for g, learner in self.learners.items()
            },
            "random": self.random.getstate(),
            "torch_random": torch.get_rng_state(),
            "numpy_random": np.random.get_state(),
            "python_random": random.getstate(),
            "cuda_random": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "best_score": self.best_score,
            "best_update": self.best_update,
            "no_improvement": self.no_improvement,
            "history": self.history,
            "actions": self.actions,
            "episode_results": self.episode_results,
            "initial": self.initial,
            "frozen": self.frozen,
        }

    def restore(self, state):
        # Reject old physical semantics before loading any mutable state.
        if state.get("physical_contract") != physical_contract(self.prepared.scenario):
            raise ValueError("resume physical/dispatch contract is incompatible")
        if state.get("task_reward_contract") != primitive(self.settings.reward.task):
            raise ValueError("resume task reward contract is incompatible")
        import numpy as np
        import torch

        if (
            state.get("action_contract") != ACTION_CONTRACT
            or state.get("observation_contract") != OBSERVATION_CONTRACT
        ):
            raise ValueError("resume decision semantics are incompatible")

        if state["scientific_sha256"] != self.prepared.scientific_sha256:
            raise ValueError("resume composition/input identity changed")
        layout = state.get("sampling_layout")
        if layout is not None and layout != {
            "num_envs": self.config.runtime.num_envs,
            "sampling_processes": self.config.runtime.sampling_processes,
        }:
            raise ValueError("resume sampling layout changed")
        if set(state["policies"]) != set(self.policies):
            raise ValueError("resume policy groups changed")
        for group, policy in self.policies.items():
            validator = getattr(policy, "validate_state_dict", None)
            if validator is not None:
                validator(state["policies"][group])
        for sampler in state.get("sampler_states", ()):
            if set(sampler) != set(self.policies):
                raise ValueError("resume sampler policy groups changed")
            for group, policy in self.policies.items():
                validator = getattr(policy, "validate_state_dict", None)
                if validator is not None:
                    validator(sampler[group])
        for key in (
            "sims",
            "episodes",
            "cursor",
            "ticks",
            "updates",
            "target_clock",
            "optimizations",
            "best_score",
            "best_update",
            "no_improvement",
            "history",
            "actions",
            "episode_results",
            "initial",
            "frozen",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
        for group, saved in state["learners"].items():
            learner = self.learners[group]
            learner.training_diagnostics = copy.deepcopy(
                state.get("learner_diagnostics", {}).get(group)
            )
            if self.settings.backend == "rllib":
                learner.set_state(saved)
            else:
                learner.policy.load_state_dict(saved["policy"])
                learner.policy.optimizer.load_state_dict(saved["optimizer"])
                learner._n_updates = saved["updates"]
        for group, saved in state["policies"].items():
            self.policies[group].load_state_dict(saved)
        if "sampler_states" in state:
            self.sampler_states = copy.deepcopy(state["sampler_states"])
        for group, saved in state["replays"].items():
            self.replays[group].load_state_dict(saved)
        self.collector.load_state_dict(state["collector"])
        self.reward_runtime.load_state_dict(state["reward"])
        self.random.setstate(state["random"])
        torch.set_rng_state(state["torch_random"])
        np.random.set_state(state["numpy_random"])
        random.setstate(state["python_random"])
        if state.get("cuda_random") is not None:
            torch.cuda.set_rng_state_all(state["cuda_random"])

    def save(self, *, best=False):
        directory = self.root / "checkpoints" / f"update-{self.updates:06d}"
        native_path(directory).mkdir(exist_ok=True)
        partners = {
            g: {
                "kind": p["implementation"]["kind"],
                "source": model_identity(p.get("resolved_model")),
                "training": g in self.settings.groups,
            }
            for g, p in json.loads(self.prepared.policies_json).items()
        }
        for group, policy in self.policies.items():
            if hasattr(policy, "network"):
                target = directory / (
                    "controllers/central" if self.central else "groups/" + group
                )
                package(
                    target,
                    policy,
                    self.prepared,
                    group,
                    self.updates,
                    partners=partners,
                )
        with native_path(directory / "continuation.pkl.tmp").open("wb") as stream:
            pickle.dump(self.state_dict(), stream, protocol=5)
        atomic_replace(
            directory / "continuation.pkl.tmp", directory / "continuation.pkl"
        )
        write_json(
            directory / "snapshot.json",
            {
                "schema": "smartsom.experiment-snapshot/v3",
                "update": self.updates,
                "physical_ticks": self.ticks,
                "scientific_sha256": self.prepared.scientific_sha256,
                "continuation_sha256": hashlib.sha256(
                    native_path(directory / "continuation.pkl").read_bytes()
                ).hexdigest(),
                "models": {
                    g: p.fingerprint()
                    for g, p in self.policies.items()
                    if hasattr(p, "fingerprint")
                },
            },
        )
        pointer = {"checkpoint": directory.name, "physical_ticks": self.ticks}
        write_json(self.root / "checkpoints/recovery.json", pointer)
        if self.config.checkpointing.save_last:
            write_json(self.root / "checkpoints/last.json", pointer)
        every = self.config.checkpointing.every_updates
        if every and self.updates % every == 0:
            write_json(
                self.root / "checkpoints" / f"periodic-{self.updates:06d}.json", pointer
            )
        recent = sorted(
            p.name for p in native_path(self.root).glob("checkpoints/update-*")
        )
        write_json(
            self.root / "checkpoints/recent.json",
            {
                "checkpoints": recent[-self.config.checkpointing.keep_last :],
                "historical_snapshots_preserved": True,
            },
        )
        if best:
            write_json(self.root / "checkpoints/best.json", pointer)
        return directory

    def report_progress(self, stage):
        from smartsom.telemetry.runtime import emit

        groups = self.collector.counts
        runtime = self.config.runtime
        event = {
            "stage": stage,
            "physical_ticks": self.ticks,
            "training_run_directory": str(self.root),
            "updates": self.updates,
            "algorithm": self.settings.algorithm.upper(),
            "bindings_summary": "; ".join(
                f"{g}={d['implementation']['kind']} ({'train' if g in self.settings.groups else 'frozen/rule'})"
                for g, d in json.loads(self.prepared.policies_json).items()
            ),
            "optimization_steps": sum(self.optimizations.values()),
            "learner_diagnostics": self.learner_diagnostics(),
            "runtime_mode": (
                f"envs={runtime.num_envs}; sampling_processes={runtime.sampling_processes}; "
                f"threads={runtime.numerical_threads}; device={runtime.device}; "
                "sampling=sequential; validation=sequential"
            ),
            "workflow": self.display_workflow,
            "validation_batches_finished": len(
                list((self.root / "logs").glob("validation-*.json"))
            ),
            "group_statistics": "; ".join(
                f"{g}: {c.get('decisions', 0)}/{c.get('training_samples', 0)}/{self.optimizations[g]}"
                for g, c in groups.items()
            ),
            "training_deliveries": sum(r["delivered"] for r in self.episode_results)
            + sum(len(sim.completed) for sim in self.sims),
        }
        if self.episode_results:
            event["episode_return"] = self.episode_results[-1]["return"]
        emit("training", event, total=self.settings.total_ticks, unit="physical ticks")

    @property
    def training_done(self):
        return (
            self.ticks >= self.settings.total_ticks
            or self.record.get("status") == "early_stopped"
        )

    def step_update(self, on_progress=None):
        """Complete sampling, learning, validation and a resumable state boundary."""
        if self.training_done:
            return copy.deepcopy(self.record)
        if hasattr(self, "_finished_result"):
            del self._finished_result
        self.record["status"] = "running"
        if not hasattr(self, "_last_progress"):
            self._last_progress = 0.0
        if (
            self.settings.record_initial
            and self.ticks == 0
            and self.updates == 0
            and not (
                native_path(self.root / "checkpoints/update-000000/continuation.pkl")
            ).exists()
        ):
            self.save()
        boundary = min(
            self.settings.total_ticks,
            self.ticks + self.settings.ticks_per_update,
        )
        self.report_progress("sampling")
        while self.ticks < boundary:
            self.advance_wave(boundary - self.ticks)
            now = time.monotonic()
            if now - self._last_progress >= 2:
                self.report_progress("sampling")
                self._last_progress = now
        if self.settings.algorithm == "ppo":
            self.report_progress("optimizing")
            self.optimize_ppo()
        self.updates += 1
        for group, fingerprint in self.frozen.items():
            if self.policies[group].fingerprint() != fingerprint:
                raise ValueError("training changed a frozen partner")
        self.record.update(
            physical_ticks=self.ticks,
            updates=self.updates,
            groups=self.collector.counts,
            optimizations=self.optimizations,
            learner_diagnostics=self.learner_diagnostics(),
        )
        write_json(self.root / "run.json", self.record)
        self.report_progress("saving")
        self.history.append(
            {
                "update": self.updates,
                "physical_ticks": self.ticks,
                "optimizations": copy.deepcopy(self.optimizations),
                "learner_diagnostics": self.learner_diagnostics(),
            }
        )
        checkpoint = self.save()
        val = self.config.validation
        best = False
        if val.enabled and self.updates % val.every_updates == 0:
            from smartsom.telemetry.runtime import emit

            emit(
                "training",
                {"validation_round": self.updates // val.every_updates},
            )
            self.report_progress("validation")
            rows = evaluate_cases(
                evaluation_recipe(self.prepared, checkpoint),
                json.loads(self.prepared.validation_json),
                validation=True,
            )
            write_json(self.root / "logs" / f"validation-{self.updates:06d}.json", rows)
            from smartsom.experiments.training_controls import (
                ValidationControls,
            )
            from smartsom.experiments.training_validation import (
                completion_delivery_return_rank,
                select_best,
            )

            controls = ValidationControls(
                **{k: getattr(val, k) for k in type(val).model_fields if k != "enabled"}
            )
            completed = [
                r
                for r in rows
                if (
                    r.get("task_window_complete", False)
                    if self.settings.reward.task is not None
                    else r["status"] == "completed"
                )
                and not r["engineering_failure"]
                and (
                    val.best_mode != "all_complete"
                    or (
                        r["makespan_complete"]
                        and r["fixed_job_makespan"] is not None
                        and isfinite(r["fixed_job_makespan"])
                    )
                )
            ]
            makespan_metric = (
                "fixed_job_makespan" if val.best_mode == "all_complete" else "makespan"
            )
            candidate = {
                "episodes": len(rows),
                "completed": len(completed),
                "successful_inputs": sorted(
                    digest([r["case_id"], r["seed"]]) for r in completed
                ),
                "metrics": {
                    "makespan": mean(r[makespan_metric] for r in completed)
                    if completed
                    and all(r[makespan_metric] is not None for r in completed)
                    else None,
                    "return": mean(r["return"] for r in completed)
                    if completed
                    else None,
                    "passing_rate": mean(r["passing_rate"] for r in completed)
                    if completed
                    and all(r["passing_rate"] is not None for r in completed)
                    else None,
                },
            }
            if val.best_mode == "completion_delivery_return":
                candidate["completion_delivery_return"] = (
                    completion_delivery_return_rank(
                        rows, len(json.loads(self.prepared.validation_json))
                    )
                )
            best, reason = select_best(candidate, self.best_score, controls)
            write_json(
                self.root / "logs" / f"selection-{self.updates:06d}.json",
                {"candidate": candidate, "selected": best, "reason": reason},
            )
            if best:
                self.best_score, self.best_update, self.no_improvement = (
                    candidate,
                    self.updates,
                    0,
                )
            else:
                self.no_improvement += 1
            self.save(best=best and self.config.checkpointing.save_best)
            self.report_progress("saving")
        if val.enabled and val.patience and self.no_improvement >= val.patience:
            self.record["status"] = "early_stopped"
        # Include the post-validation best/patience/RNG state even if validation
        # is disabled. Adaptive continuation consumes only this full boundary.
        self.save()
        write_json(self.root / "run.json", self.record)
        if on_progress:
            on_progress(copy.deepcopy(self.record))
        return copy.deepcopy(self.record)

    def finish(self):
        """Write final training evidence without advancing an update."""
        from smartsom.api import TrainingResult

        if hasattr(self, "_finished_result"):
            return self._finished_result
        if self.training_done and self.record.get("status") != "early_stopped":
            self.record["status"] = "completed"
        self.record.update(
            changed_weights={
                g: self.policies[g].fingerprint() != self.initial[g]
                for g in self.initial
            },
            actual_optimization_steps=self.optimizations,
            frozen_partners_unchanged=True,
        )
        write_json(self.root / "run.json", self.record)
        write_json(self.root / "evidence/actions.json", self.actions)
        write_json(self.root / "reports/training.json", self.history)
        last = (
            checkpoint_path(self.root, "last")
            if (native_path(self.root / "checkpoints/last.json")).exists()
            else None
        )
        best = (
            checkpoint_path(self.root, "best")
            if (native_path(self.root / "checkpoints/best.json")).exists()
            else None
        )
        self._finished_result = TrainingResult(
            self.root,
            self.root,
            last,
            best,
            self.ticks,
            self.updates,
            sum(self.optimizations.values()) if self.settings.algorithm == "ppo" else 0,
            self.record["status"],
        )

        return self._finished_result

    def close(self):
        if self.executor:
            self.executor.shutdown(cancel_futures=True)
            self.executor = None

    def execute(self, on_progress=None, *, stop_after_updates=None):
        try:
            self.report_progress("initializing")
            while not self.training_done:
                boundary(self.root)
                self.step_update(on_progress)
                boundary(self.root)
                if (
                    stop_after_updates is not None
                    and self.updates >= stop_after_updates
                ):
                    self.record["status"] = "interrupted"
                    break
            return self.finish()
        except BaseException as exc:
            self.record.update(
                status="interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                failure={"exception": type(exc).__name__, "message": str(exc)},
            )
            write_json(self.root / "run.json", self.record)
            raise
        finally:
            self.close()


def train(prepared, *, on_progress=None, initialize_from=None):
    if initialize_from is not None:
        raise ValueError(
            "v3 initialization belongs in each strategy policy model selector"
        )
    root, record, prepared = allocate(prepared, "training")
    try:
        return TrainingSession(prepared, root, record).execute(on_progress)
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure={"exception": type(exc).__name__, "message": str(exc)},
        )
        write_json(root / "run.json", record)
        exc.run_dir = root
        raise


def resume(source, *, on_progress=None):
    root = Path(source).resolve()
    record = json.loads((native_path(root / "run.json")).read_text(encoding="utf-8"))
    if record.get("implementation_sha256") != implementation_identity():
        raise ValueError(
            "resume implementation identity changed; initialize a new experiment instead"
        )
    current = source_identity()
    for backend in ("torch", "ray", "stable-baselines3", "sb3-contrib"):
        if record["source"]["packages"].get(backend) != current["packages"].get(
            backend
        ):
            raise ValueError("resume backend version changed")
    prepared = prepared_from_run(root)
    recovery = json.loads(
        (native_path(root / "checkpoints/recovery.json")).read_text(encoding="utf-8")
    )["checkpoint"]
    snapshot = checkpoint_path(root, recovery)
    metadata = json.loads(
        (native_path(snapshot / "snapshot.json")).read_text(encoding="utf-8")
    )
    if (
        hashlib.sha256(
            (native_path(snapshot / "continuation.pkl")).read_bytes()
        ).hexdigest()
        != metadata["continuation_sha256"]
    ):
        raise ValueError("continuation state hash mismatch")
    # Complete local experiment snapshots are trusted Python continuation state,
    # unlike portable inference ZIPs, which use weights_only tensor loading.
    with native_path(snapshot / "continuation.pkl").open("rb") as stream:
        state = pickle.load(stream)
    from smartsom.telemetry.runtime import bind, configure_workflow

    bind(root)
    configure_workflow(prepared, "training")
    session = TrainingSession(prepared, root, record)
    session.restore(state)
    record["status"] = "running"
    return session.execute(on_progress)


def export(source, destination, *, group=None, selection="last", kind="model"):
    import zipfile

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if native_path(destination).exists():
        raise FileExistsError(destination)
    prepared = prepared_from_run(source)
    if kind == "experiment":
        native_path(destination.parent).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            native_path(destination), "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(native_path(source).rglob("*")):
                if native_path(path).is_file():
                    archive.write(
                        native_path(path), str(path.relative_to(native_path(source)))
                    )
        return destination
    snapshot = checkpoint_path(source, selection)
    central = bool(json.loads(prepared.composition_json).get("controller"))
    if central:
        if group is not None:
            raise ValueError("central controllers export as a whole; omit --group")
        model = snapshot / "controllers/central"
    else:
        if group is None:
            raise ValueError("resource export requires --group")
        if group not in json.loads(prepared.policies_json):
            raise ValueError("unknown strategy group")
        model = snapshot / "groups" / group
    if not (native_path(model / "model.json")).exists():
        raise ValueError("rules have no learned component to export")
    native_path(destination.parent).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        native_path(destination), "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ("model.json", "weights.pt", "encoder.json"):
            archive.write(native_path(model / name), name)
    return destination
