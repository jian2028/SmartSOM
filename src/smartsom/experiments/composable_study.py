"""Prepare and execute frozen composable H/V studies through ordinary v3 APIs."""

import csv
import hashlib
import json
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import Field, model_validator

from smartsom.config.codec import (
    ConfigurationError,
    canonical_json,
    digest,
    primitive,
    read_model,
)
from smartsom.config.experiment import LoggingOptions
from smartsom.config.experiment_v3 import model_location, prepare_v3, read_document
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.models import StrictModel
from smartsom.config.policies import ModelSelector
from smartsom.config.production import named_seed
from smartsom.config.travel_time import matrix_summary
from smartsom.experiments.batch import exclusive_lock
from smartsom.experiments.composable import (
    archive_inputs,
    evaluate_cases,
    evaluation_recipe,
    implementation_identity,
    prepared_from_run,
    summarize,
)
from smartsom.experiments.evidence import source_identity
from smartsom.telemetry.runtime import CURRENT, bind, operation, worker_output
from smartsom.telemetry.study_progress import StudyWork
from smartsom.workloads.job_content import ContentRecipe, generate_content


class StudyRecipe(StrictModel):
    schema_id: Literal["smartsom.composable-study/v1"] = Field(alias="schema")
    factories: dict[str, str]
    workload: str
    transport: dict[str, str]
    compositions: dict[str, str]
    algorithms: dict[str, str]
    logging: LoggingOptions = Field(default_factory=LoggingOptions)
    training_seed: int = Field(default=101, ge=0, lt=2**64)
    validation_seed: int = Field(default=303, ge=0, lt=2**64)
    evaluation_seed: int = Field(default=202, ge=0, lt=2**64)
    max_concurrent: int = Field(default=8, gt=0)
    total_ticks: int = Field(default=16384, gt=0)
    ticks_per_update: int = Field(default=256, gt=0)
    max_ticks: int = Field(default=4096, gt=0)
    validation_every_updates: int = Field(default=4, gt=0)
    validation_cases: int = Field(default=5, gt=0)
    evaluation_cases: int = Field(default=5, gt=0)
    controls: tuple[Literal["initial", "rule", "random"], ...] = (
        "initial",
        "rule",
        "random",
    )

    @model_validator(mode="after")
    def valid(self):
        if set(self.factories) != {"h0", "h1"} or set(self.transport) != {
            "auto",
            "zero",
        }:
            raise ValueError("Small study requires h0/h1 and auto/zero transport cases")
        if set(self.compositions) != {"ppo", "dqn"} or set(self.algorithms) != {
            "ppo",
            "dqn",
        }:
            raise ValueError("provide PPO and DQN configurations")
        if len({self.training_seed, self.validation_seed, self.evaluation_seed}) != 3:
            raise ValueError("training, validation and test seed roots must differ")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("duplicate study control")
        return self


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n")
    temporary.replace(path)


def _yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(primitive(value), allow_unicode=True, sort_keys=False)
    )


def heterogeneity(factory, baseline):
    """Nominal rate and capacity-weighted quality invariants from the model."""
    before = primitive(baseline)
    after = primitive(factory)
    for doc in (before, after):
        doc.pop("name", None)
        for machine in doc["machines"]:
            machine.pop("name", None)
            machine.pop("quality_modes", None)
            machine.pop("processing_rate_multiplier", None)
    if before != after:
        raise ConfigurationError(
            "H may change performance only, not identity, geometry or compatibility"
        )
    speed = []
    quality = []
    tables = []
    originals = {m.machine_id: m for m in baseline.machines}
    for kind in sorted(factory.operation_types):
        machines = [m for m in factory.machines if kind in m.operation_types]
        for mode_id in ("slow", "normal", "fast"):
            reference = [
                x
                for x in originals[machines[0].machine_id].quality_modes
                if x.quality_mode_id == mode_id
            ][0]
            base_rate = Fraction(
                originals[machines[0].machine_id].processing_rate_multiplier
            ) / Fraction(reference.time_scale)
            base_q = 1 - Fraction(reference.error_rate)
            ratios = []
            qs = []
            for machine in machines:
                mode = next(
                    x for x in machine.quality_modes if x.quality_mode_id == mode_id
                )
                ratios.append(
                    Fraction(machine.processing_rate_multiplier)
                    / Fraction(mode.time_scale)
                    / base_rate
                )
                qs.append(1 - Fraction(mode.error_rate))
            if (
                sum(ratios) != len(machines)
                or sum(c * q for c, q in zip(ratios, qs, strict=True))
                != len(machines) * base_q
            ):
                raise ConfigurationError(
                    "H violates nominal rate or capacity-weighted quality conservation"
                )
            speed.append(sum((v - 1) ** 2 for v in ratios) / len(machines))
            quality.append(
                sum(
                    c * (q - base_q) ** 2 / Fraction("0.01") ** 2
                    for c, q in zip(ratios, qs, strict=True)
                )
                / sum(ratios)
            )
            tables.append(
                {
                    "operation": kind,
                    "mode": mode_id,
                    "rate_ratios": list(map(float, ratios)),
                    "qualities": list(map(float, qs)),
                }
            )
    hs = float(sum(speed) / len(speed)) ** 0.5
    hq = float(sum(quality) / len(quality)) ** 0.5
    return {
        "speed": hs,
        "quality": hq,
        "H": (0.5 * hs**2 + 0.5 * hq**2) ** 0.5,
        "tables": tables,
    }


def rounding_diagnostics(factory, pool):
    import math

    losses = []
    collapsed = 0
    pairs = 0
    for job in pool:
        for kind, ticks in job["spec"]:
            machines = [m for m in factory.machines if kind in m.operation_types]
            for mode_id in ("slow", "normal", "fast"):
                durations = []
                rates = []
                for machine in machines:
                    mode = next(
                        m for m in machine.quality_modes if m.quality_mode_id == mode_id
                    )
                    exact = (
                        Fraction(ticks)
                        * Fraction(mode.time_scale)
                        / Fraction(machine.processing_rate_multiplier)
                    )
                    duration = max(1, math.ceil(exact))
                    losses.append(float(Fraction(duration) / exact - 1))
                    durations.append(duration)
                    rates.append(machine.processing_rate_multiplier)
                if len(set(rates)) > 1:
                    pairs += 1
                    collapsed += int(len(set(durations)) == 1)
    return {
        "mean_relative_ceiling_loss": sum(losses) / len(losses),
        "max_relative_ceiling_loss": max(losses),
        "different_rate_pairs": pairs,
        "pairs_with_equal_integer_duration": collapsed,
    }


def _copy_composition(source, target, algorithm):
    document = read_document(source)
    for group, ref in document["groups"].items():
        policy_source = (source.parent / ref["policy"]).resolve()
        policy = read_document(policy_source)
        if policy["implementation"]["kind"] == "model":
            model = policy["implementation"]["model"]
            model["source"] = str((policy_source.parent / model["source"]).resolve())
        policy_target = target.parent.parent / "policies" / f"{algorithm}_{group}.yaml"
        _yaml(policy_target, policy)
        ref["policy"] = "../policies/" + policy_target.name
    matching_source = (source.parent / document["pickup_matching"]).resolve()
    matching_target = target.parent.parent / "rules" / f"{algorithm}_pickup.yaml"
    _yaml(matching_target, read_document(matching_source))
    document["pickup_matching"] = "../rules/" + matching_target.name
    _yaml(target, document)


def source_inputs(config, recipe):
    """Hash referenced authoring files and resolve model aliases before reuse."""
    files = {Path(config).resolve()}
    models = []
    for paths in (recipe.factories, recipe.transport, recipe.algorithms):
        files.update((config.parent / ref).resolve() for ref in paths.values())
    files.add((config.parent / recipe.workload).resolve())
    for ref in recipe.compositions.values():
        composition_path = (config.parent / ref).resolve()
        files.add(composition_path)
        composition = read_document(composition_path)
        files.add((composition_path.parent / composition["pickup_matching"]).resolve())
        for group in composition["groups"].values():
            policy_path = (composition_path.parent / group["policy"]).resolve()
            files.add(policy_path)
            policy = read_document(policy_path)
            if policy["implementation"]["kind"] == "model":
                selector = dict(policy["implementation"]["model"])
                selector["source"] = str(
                    (policy_path.parent / selector["source"]).resolve()
                )
                models.append(model_location(ModelSelector(**selector)))
    return {
        "files": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
        },
        "models": models,
    }


def prepare_study(config, output):
    from smartsom import api

    config = Path(config).resolve()
    output = Path(output).resolve()
    recipe, recipe_hash = read_model(config, StudyRecipe)
    identity = implementation_identity()
    inputs = source_inputs(config, recipe)
    if output.exists():
        plan = _load(output)
        if plan["recipe_sha256"] != recipe_hash or plan.get("source_inputs") != inputs:
            raise ConfigurationError(
                "output already contains a different study; choose a new output directory"
            )
        return show_study(output)
    content_path = (config.parent / recipe.workload).resolve()
    content, _ = read_model(content_path, ContentRecipe)
    # Compute all datasets before creating a visible prepared study.
    datasets = {}
    splits = (
        ("training", recipe.training_seed, 1),
        ("validation", recipe.validation_seed, recipe.validation_cases),
        ("test", recipe.evaluation_seed, recipe.evaluation_cases),
    )
    for split, seed, count in splits:
        for index in range(count):
            label = f"{split}_{index:03d}"
            datasets[label] = generate_content(
                content, named_seed(seed, f"pool:{index}"), label
            )
    factories = {
        name: load_factory_design_file(config.parent / path)[0]
        for name, path in recipe.factories.items()
    }
    h = {
        name: heterogeneity(doc.factory, factories["h0"].factory)
        for name, doc in factories.items()
    }
    for name, doc in factories.items():
        h[name]["rounding"] = rounding_diagnostics(
            doc.factory, datasets["training_000"]["pool"]
        )
    all_types = {k for template in content.templates for k in template.route}
    if not all_types <= set(factories["h0"].factory.operation_types):
        raise ConfigurationError(
            "content recipe references unavailable operation types"
        )
    if (content.windows - 1) * content.window_ticks + (
        content.jobs_per_window - 1
    ) * content.window_ticks // content.jobs_per_window >= recipe.max_ticks:
        raise ConfigurationError(
            "episode limit excludes declared original arrival slots"
        )
    output.mkdir(parents=True)
    try:
        for folder in ("config", "snapshots", "datasets", "experiments", "controls"):
            (output / folder).mkdir()
        for name, doc in factories.items():
            _yaml(output / "config/factories" / (name + ".yaml"), doc)
        for name, path in recipe.transport.items():
            _yaml(
                output / "config/transport" / (name + ".yaml"),
                read_document(config.parent / path),
            )
        for algorithm, path in recipe.compositions.items():
            _copy_composition(
                (config.parent / path).resolve(),
                output / "config/compositions" / (algorithm + ".yaml"),
                algorithm,
            )
            _yaml(
                output / "config/algorithms" / (algorithm + ".yaml"),
                read_document(config.parent / recipe.algorithms[algorithm]),
            )
        for label, dataset in datasets.items():
            _json(output / "datasets" / (label + ".json"), dataset)
            for level, row in dataset["levels"].items():
                _yaml(
                    output / "config/workloads" / (label + "_" + level + ".yaml"),
                    row["workload"],
                )
        entries = []
        for h_name in recipe.factories:
            for level in ("low", "mid", "high"):
                for transport in recipe.transport:
                    scenario_paths = {}
                    for label in datasets:
                        name = f"{h_name}_{level}_{transport}_{label}"
                        target = output / "config/scenarios" / (name + ".yaml")
                        _yaml(
                            target,
                            {
                                "schema": "smartsom.scenario/v2",
                                "factory": "../factories/" + h_name + ".yaml",
                                "workload": "../workloads/"
                                + label
                                + "_"
                                + level
                                + ".yaml",
                                "mode": "finite",
                                "tick_limit": recipe.max_ticks,
                                "processing_rounding": "ceil",
                                "transport": {
                                    "mode": "travel_time_matrix",
                                    "matrix": "../transport/" + transport + ".yaml",
                                },
                            },
                        )
                        scenario_paths[label] = "../scenarios/" + target.name
                    for algorithm in recipe.compositions:
                        name = f"{h_name}_{level}_{algorithm}_{transport}"
                        target = output / "config/runs" / (name + ".yaml")
                        config_doc = {
                            "schema": "smartsom.experiment-config/v3",
                            "scenario": scenario_paths["training_000"],
                            "composition": "../compositions/" + algorithm + ".yaml",
                            "seed": recipe.training_seed,
                            "training": {
                                "mode": "resource",
                                "backend": "rllib",
                                "algorithm": algorithm,
                                "parameters": "../algorithms/" + algorithm + ".yaml",
                                "groups": ["machine", "buffer", "dispatcher"],
                                "total_ticks": recipe.total_ticks,
                                "ticks_per_update": recipe.ticks_per_update,
                                "max_ticks": recipe.max_ticks,
                                "record_initial": True,
                            },
                            "runtime": {
                                "num_envs": 1,
                                "sampling_processes": 0,
                                "numerical_threads": 1,
                                "device": "cpu",
                            },
                            "validation": {
                                "enabled": True,
                                "every_updates": recipe.validation_every_updates,
                                "seed": recipe.validation_seed,
                                "replications": 1,
                                "scenarios": [
                                    scenario_paths[k]
                                    for k in datasets
                                    if k.startswith("validation_")
                                ],
                            },
                            "evaluation": {
                                "checkpoint": "last",
                                "seed": recipe.evaluation_seed,
                                "replications": 1,
                                "scenarios": [
                                    scenario_paths[k]
                                    for k in datasets
                                    if k.startswith("test_")
                                ],
                            },
                            "logging": {
                                "title": recipe.logging.title,
                                "task_title": recipe.logging.task_title,
                            },
                            "output": {
                                "root": str(output / "experiments"),
                                "name": name,
                            },
                        }
                        _yaml(target, config_doc)
                        prepared = prepare_v3(api.load_config(target))
                        # Archive model partners if custom compositions use existing weights.
                        (output / "snapshots" / name / "config").mkdir(parents=True)
                        prepared = archive_inputs(output / "snapshots" / name, prepared)
                        snap = output / "snapshots" / name / "config/prepared.json"
                        entries.append(
                            {
                                "id": name,
                                "H_case": h_name,
                                "H": h[h_name]["H"],
                                "V_case": level,
                                "V": datasets["training_000"]["levels"][level]["V"],
                                "algorithm": algorithm,
                                "training_seed": recipe.training_seed,
                                "transport": transport,
                                "matrix": matrix_summary(
                                    prepared.scenario.transport_matrix
                                ),
                                "config": str(target.relative_to(output)),
                                "snapshot": str(snap.relative_to(output)),
                                "snapshot_sha256": digest(json.loads(snap.read_text())),
                                "scientific_sha256": prepared.scientific_sha256,
                            }
                        )
        plan = {
            "schema": "smartsom.composable-study-plan/v1",
            "recipe": primitive(recipe),
            "recipe_sha256": recipe_hash,
            "source_inputs": inputs,
            "implementation_sha256": identity,
            "source": source_identity(),
            "development_pilot": True,
            "H": h,
            "content_recipe": primitive(content),
            "datasets": {
                label: {
                    "pool_sha256": data["pool_sha256"],
                    "V": {k: v["V"] for k, v in data["levels"].items()},
                    "reference_work": data["reference_work"],
                }
                for label, data in datasets.items()
            },
            "entries": entries,
        }
        _json(output / "plan.json", plan)
        _json(
            output / "study.json",
            {
                "plan_sha256": digest(plan),
                "entries": {},
                "controls": {},
                "status": "prepared",
            },
        )
    except BaseException:
        # Keep diagnostics; never overwrite or automatically remove a partial bundle.
        _json(
            output / "prepare-failure.json",
            {"status": "failed", "implementation_sha256": identity},
        )
        raise
    return show_study(output)


def _load(directory):
    directory = Path(directory).resolve()
    try:
        plan = json.loads((directory / "plan.json").read_text())
        state = json.loads((directory / "study.json").read_text())
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"invalid prepared study: {directory}: {exc}") from exc
    if digest(plan) != state["plan_sha256"]:
        raise ConfigurationError("frozen study plan changed")
    live_source = source_identity()
    if (
        any(live_source[key] != plan["source"][key] for key in ("python", "packages"))
        or live_source["git"]["commit"] != plan["source"]["git"]["commit"]
    ):
        raise ConfigurationError(
            "source or dependency identity changed; prepare a new study"
        )
    if implementation_identity() != plan["implementation_sha256"]:
        raise ConfigurationError("implementation changed; prepare a new study")
    for row in plan["entries"]:
        path = (directory / row["snapshot"]).resolve()
        if (
            not path.is_relative_to(directory)
            or digest(json.loads(path.read_text())) != row["snapshot_sha256"]
        ):
            raise ConfigurationError("frozen child snapshot changed")
    return plan


def show_study(directory):
    directory = Path(directory).resolve()
    plan = _load(directory)
    state = json.loads((directory / "study.json").read_text())
    recipe = plan["recipe"]
    updates = (recipe["total_ticks"] + recipe["ticks_per_update"] - 1) // recipe[
        "ticks_per_update"
    ]
    return {
        "directory": str(directory),
        "development_pilot": True,
        "status": state["status"],
        "max_concurrent": recipe.get("max_concurrent", 1),
        "sampling": "one sequential environment per experiment",
        "validation": "sequential cases per experiment",
        "training_runs": len(plan["entries"]),
        "training_ticks": len(plan["entries"]) * recipe["total_ticks"],
        "validation_episodes": len(plan["entries"])
        * (updates // recipe["validation_every_updates"])
        * recipe["validation_cases"],
        "model_test_episodes": len(plan["entries"]) * recipe["evaluation_cases"],
        "H": plan["H"],
        "datasets": plan["datasets"],
        "entries": [
            dict(
                row, status=state["entries"].get(row["id"], {}).get("status", "pending")
            )
            for row in plan["entries"]
        ],
    }


def _prepared(directory, row):
    return prepared_from_run(directory / Path(row["snapshot"]).parents[1])


def _rule_recipe(prepared, random=False):
    declarations = json.loads(prepared.policies_json)
    names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": "nearest",
        "mover": "automatic_travel",
    }
    for declaration in declarations.values():
        role = declaration["role"]
        declaration.pop("resolved_model", None)
        declaration["implementation"] = {
            "kind": "rule",
            "name": "random" if random and role != "mover" else names[role],
            "parameters": {},
        }
    return replace(prepared, policies_json=canonical_json(declarations))


@contextmanager
def _control_lock(path):
    """Paired experiments share controls; exactly one process computes a cache."""
    import fcntl

    with path.open("a+") as stream:
        from smartsom.experiments.control import boundary

        while True:
            boundary()
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@worker_output("worker_dir")
@operation("study-entry")
def _study_worker(directory, row, plan, entry, retry_failed, worker_dir):
    from smartsom import api

    directory, worker_dir = Path(directory), Path(worker_dir)
    bind(worker_dir, row["id"])
    entry["started_at"] = time.time()
    entry["pid"] = os.getpid()
    session = CURRENT.get()
    state = {"entries": {row["id"]: entry}, "controls": {}}
    prepared = _prepared(directory, row)

    def persist():
        _json(worker_dir / "entry.json", state)

    try:
        # Adopt compatible manual single-run work rather than retraining it.
        source = entry.get("run_dir")
        if not source:
            matches = []
            for run in (directory / "experiments").glob("*/run.json"):
                record = json.loads(run.read_text())
                if (
                    record.get("scientific_sha256") == prepared.scientific_sha256
                    and record.get("implementation_sha256")
                    == plan["implementation_sha256"]
                    and record.get("kind") == "training"
                ):
                    matches.append(run.parent)
            if matches:
                source = str(sorted(matches)[-1])
                entry["run_dir"] = source
        try:
            if source:
                record = json.loads((Path(source) / "run.json").read_text())
                if (
                    record.get("scientific_sha256") != prepared.scientific_sha256
                    or record.get("implementation_sha256")
                    != plan["implementation_sha256"]
                ):
                    raise ConfigurationError(
                        "saved child identity differs from frozen study"
                    )
                if record["status"] not in ("completed", "early_stopped"):
                    if (Path(source) / "checkpoints/recovery.json").exists():
                        api.resume(source)
                    else:
                        result = api.train_prepared(prepared)
                        source = str(result.run_dir)
                        entry["run_dir"] = source
            else:
                result = api.train_prepared(prepared)
                source = str(result.run_dir)
                entry["run_dir"] = source
            persist()
            evaluated = api.evaluate(source)
            entry["evaluation_dir"] = str(evaluated.run_dir)
            persist()
            if any(
                r["engineering_failure"]
                for r in json.loads((evaluated.run_dir / "run.json").read_text())[
                    "results"
                ]
            ):
                raise RuntimeError("final evaluation has engineering failures")
            frozen = prepared_from_run(source)
            cases = json.loads(frozen.evaluation_json)
            if "initial" in plan["recipe"]["controls"]:
                control = directory / "controls" / (row["id"] + "_initial.json")
                if control.exists() and retry_failed:
                    cached = json.loads(control.read_text())
                    if any(r["engineering_failure"] for r in cached):
                        control.replace(
                            control.with_name(
                                control.stem + ".failed-" + uuid4().hex + ".json"
                            )
                        )
                if not control.exists():
                    initial = Path(source) / "checkpoints/update-000000"
                    rows = evaluate_cases(
                        evaluation_recipe(frozen, initial),
                        cases,
                        label="initial control",
                    )
                    _json(control, rows)
                rows = json.loads(control.read_text())
                if any(r["engineering_failure"] for r in rows):
                    raise RuntimeError("initial-model control has engineering failures")
            for name in ("rule", "random"):
                if name not in plan["recipe"]["controls"]:
                    continue
                key = f"{row['H_case']}_{row['V_case']}_{row['transport']}_{name}"
                session.update(
                    "evaluation",
                    {"stage": "waiting for shared control", "evaluation_kind": name},
                )
                with _control_lock(directory / "controls" / (key + ".lock")):
                    if key not in state["controls"]:
                        control = directory / "controls" / (key + ".json")
                        if control.exists() and retry_failed:
                            cached = json.loads(control.read_text())
                            if any(r["engineering_failure"] for r in cached):
                                control.replace(
                                    control.with_name(
                                        control.stem
                                        + ".failed-"
                                        + uuid4().hex
                                        + ".json"
                                    )
                                )
                        if control.exists():
                            rows = json.loads(control.read_text())
                        else:
                            rows = evaluate_cases(
                                _rule_recipe(frozen, name == "random"),
                                cases,
                                label=name + " control",
                            )
                        _json(control, rows)
                        if any(r["engineering_failure"] for r in rows):
                            raise RuntimeError(
                                f"{name} control has engineering failures"
                            )
                        state["controls"][key] = str(control.relative_to(directory))
            entry["status"] = "completed"
            entry.pop("error", None)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if getattr(exc, "run_dir", None):
                entry["run_dir"] = str(exc.run_dir)
    except KeyboardInterrupt:
        entry["status"] = "interrupted"
        session.status = "interrupted"
        persist()
        raise
    finally:
        entry["ended_at"] = time.time()
        persist()
    session.status = entry["status"]


def _collect_worker_progress(session, worker_dir, row, entry, total):
    if session is None:
        return
    event = {
        "stage": entry.get("status", "starting"),
        "study_case": {
            "algorithm": row["algorithm"].upper(),
            "H": row["H_case"],
            "V": row["V_case"],
            "travel": row["transport"],
            "seed": row.get("training_seed"),
        },
    }
    try:
        snapshot = json.loads((worker_dir / "logs/progress.json").read_text())
        tasks = snapshot["tasks"]
        event["study_case"]["last_activity"] = snapshot["updated_at"]
        training = next((r for r in tasks if r["id"] == "training"), None)
        latest = max(tasks, key=lambda r: r.get("updated_at", 0), default=None)
        if training:
            event.update(training["values"])
        if latest:
            event.update(latest["values"])
            event["stage"] = latest["stage"]
        event["context"] = (
            f"{row['algorithm'].upper()} | H={row['H_case']} V={row['V_case']} "
            f"travel={row['transport']} | pid={entry.get('pid', 'starting')} envs=1 sampling=sequential validation=sequential "
            f"| last activity {time.strftime('%H:%M:%S', time.localtime(snapshot['updated_at']))}"
        )
    except (OSError, ValueError, KeyError):
        event["context"] = (
            f"{row['algorithm'].upper()} | H={row['H_case']} V={row['V_case']} travel={row['transport']} | starting"
        )
    final = entry.get("status") in {"completed", "failed", "interrupted"}
    if final:
        event.update(status=entry["status"], stage=entry["status"])
        if entry.get("error"):
            event["reason"] = entry["error"]
    session.update(row["id"], event, total=total, unit="physical ticks", final=final)


@operation("study")
def run_study(directory, *, retry_failed=False):
    """Bounded independent processes; parent alone owns the manifest and terminal."""
    directory = Path(directory).resolve()
    from smartsom.experiments.control import StopRequested, boundary

    plan = _load(directory)
    concurrency = plan["recipe"].get("max_concurrent", 1)
    if concurrency == 1:
        return _run_serial_study(directory, retry_failed=retry_failed)
    bind(directory)
    session = CURRENT.get()
    from smartsom.telemetry.workflow import describe_study

    session.configure(describe_study(directory, plan))
    session.total_tasks = len(plan["entries"])
    work = StudyWork(directory, plan)
    workers = directory / "workers"
    workers.mkdir(exist_ok=True)
    with exclusive_lock(directory / "study.lock"):
        state = json.loads((directory / "study.json").read_text())
        state["status"] = "running"
        pending, active = [], {}
        for row in plan["entries"]:
            entry = state["entries"].setdefault(row["id"], {})
            if entry.get("status") == "completed" or (
                entry.get("status") == "failed" and not retry_failed
            ):
                _collect_worker_progress(
                    session,
                    workers / row["id"],
                    row,
                    entry,
                    plan["recipe"]["total_ticks"],
                )
                continue
            entry["status"] = "queued"
            session.update(
                row["id"],
                {"status": "queued", "stage": "queued"},
                total=plan["recipe"]["total_ticks"],
                unit="physical ticks",
            )
            pending.append(row)
        _json(directory / "study.json", state)
        context = multiprocessing.get_context("spawn")

        def collect(row, *, ended=False):
            worker_dir = workers / row["id"]
            path = worker_dir / "entry.json"
            if path.exists():
                data = json.loads(path.read_text())
                state["entries"][row["id"]].update(data["entries"][row["id"]])
                state["controls"].update(data["controls"])
            entry = state["entries"][row["id"]]
            if ended and entry.get("status") == "running":
                entry.update(
                    status="failed",
                    error="worker exited without a final result; see worker.log",
                )
            _collect_worker_progress(
                session, worker_dir, row, entry, plan["recipe"]["total_ticks"]
            )

        try:
            while pending or active:
                boundary(directory)
                while pending and len(active) < concurrency:
                    row = pending.pop(0)
                    worker_dir = workers / row["id"]
                    worker_dir.mkdir(exist_ok=True)
                    entry = state["entries"][row["id"]]
                    entry["status"] = "running"
                    _json(
                        worker_dir / "entry.json",
                        {"entries": {row["id"]: entry}, "controls": {}},
                    )
                    process = context.Process(
                        target=_study_worker,
                        args=(
                            directory,
                            row,
                            plan,
                            dict(entry),
                            retry_failed,
                            worker_dir,
                        ),
                    )
                    process.start()
                    entry["pid"] = process.pid
                    active[row["id"]] = (process, row)
                    _json(directory / "study.json", state)
                session.notice = f"Parallel experiments: {len(active)}/{concurrency}; queued: {len(pending)}. Each experiment: 1 env, sequential sampling/validation, CPU, 1 thread."
                with session.batch_updates():
                    for key, (process, row) in list(active.items()):
                        ended = not process.is_alive()
                        collect(row, ended=ended)
                        if ended:
                            process.join()
                            del active[key]
                            _json(directory / "study.json", state)
                            _report(directory, plan, state)
                    session.overview = work.overview(session.tasks, state)
                    session.publish()
                time.sleep(0.5)
        except BaseException as exc:
            if isinstance(exc, StopRequested):
                # Workers see the same ancestor request at their own safe boundary.
                # The stop command's explicit --force owns timeout escalation.
                for process, row in active.values():
                    process.join()
                    collect(row)
                state["status"] = "interrupted"
                _json(directory / "study.json", state)
                raise
            for process, _ in active.values():
                if process.is_alive():
                    try:
                        os.kill(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            for process, row in active.values():
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                collect(row)
                if state["entries"][row["id"]].get("status") == "running":
                    state["entries"][row["id"]]["status"] = "interrupted"
            state["status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            _json(directory / "study.json", state)
            _report(directory, plan, state)
            session.overview = work.overview(session.tasks, state)
            raise
        state["status"] = (
            "completed"
            if all(e.get("status") == "completed" for e in state["entries"].values())
            else "failed"
        )
        session.status = state["status"]
        session.overview = work.overview(session.tasks, state)
        session.notice = f"Parallel experiments: 0/{concurrency}; finished. Each experiment used one sequential environment."
        _json(directory / "study.json", state)
        _report(directory, plan, state)
    return show_study(directory)


def _run_serial_study(directory, *, retry_failed=False):
    """Serial, restartable execution. Every child uses the ordinary v3 lifecycle."""
    from smartsom import api

    directory = Path(directory).resolve()
    from smartsom.experiments.control import boundary

    bind(directory)
    plan = _load(directory)
    with exclusive_lock(directory / "study.lock"):
        state = json.loads((directory / "study.json").read_text())
        state["status"] = "running"
        _json(directory / "study.json", state)
        try:
            for row in plan["entries"]:
                boundary(directory)
                entry = state["entries"].setdefault(row["id"], {})
                prepared = _prepared(directory, row)
                if entry.get("status") == "completed":
                    continue
                if entry.get("status") == "failed" and not retry_failed:
                    continue
                entry["status"] = "running"
                _json(directory / "study.json", state)
                # Adopt compatible manual single-run work rather than retraining it.
                source = entry.get("run_dir")
                if not source:
                    matches = []
                    for run in (directory / "experiments").glob("*/run.json"):
                        record = json.loads(run.read_text())
                        if (
                            record.get("scientific_sha256")
                            == prepared.scientific_sha256
                            and record.get("implementation_sha256")
                            == plan["implementation_sha256"]
                            and record.get("kind") == "training"
                        ):
                            matches.append(run.parent)
                    if matches:
                        source = str(sorted(matches)[-1])
                        entry["run_dir"] = source
                try:
                    if source:
                        record = json.loads((Path(source) / "run.json").read_text())
                        if (
                            record.get("scientific_sha256")
                            != prepared.scientific_sha256
                            or record.get("implementation_sha256")
                            != plan["implementation_sha256"]
                        ):
                            raise ConfigurationError(
                                "saved child identity differs from frozen study"
                            )
                        if record["status"] not in ("completed", "early_stopped"):
                            if (Path(source) / "checkpoints/recovery.json").exists():
                                api.resume(source)
                            else:
                                result = api.train_prepared(prepared)
                                source = str(result.run_dir)
                                entry["run_dir"] = source
                    else:
                        result = api.train_prepared(prepared)
                        source = str(result.run_dir)
                        entry["run_dir"] = source
                    _json(directory / "study.json", state)
                    evaluated = api.evaluate(source)
                    entry["evaluation_dir"] = str(evaluated.run_dir)
                    if any(
                        r["engineering_failure"]
                        for r in json.loads(
                            (evaluated.run_dir / "run.json").read_text()
                        )["results"]
                    ):
                        raise RuntimeError("final evaluation has engineering failures")
                    frozen = prepared_from_run(source)
                    cases = json.loads(frozen.evaluation_json)
                    if "initial" in plan["recipe"]["controls"]:
                        control = directory / "controls" / (row["id"] + "_initial.json")
                        if control.exists() and retry_failed:
                            cached = json.loads(control.read_text())
                            if any(r["engineering_failure"] for r in cached):
                                control.replace(
                                    control.with_name(
                                        control.stem
                                        + ".failed-"
                                        + uuid4().hex
                                        + ".json"
                                    )
                                )
                        if not control.exists():
                            initial = Path(source) / "checkpoints/update-000000"
                            rows = evaluate_cases(
                                evaluation_recipe(frozen, initial),
                                cases,
                                label="initial control",
                            )
                            _json(control, rows)
                        rows = json.loads(control.read_text())
                        if any(r["engineering_failure"] for r in rows):
                            raise RuntimeError(
                                "initial-model control has engineering failures"
                            )
                    for name in ("rule", "random"):
                        if name not in plan["recipe"]["controls"]:
                            continue
                        key = (
                            f"{row['H_case']}_{row['V_case']}_{row['transport']}_{name}"
                        )
                        if key not in state["controls"]:
                            control = directory / "controls" / (key + ".json")
                            rows = evaluate_cases(
                                _rule_recipe(frozen, name == "random"), cases
                            )
                            _json(control, rows)
                            if any(r["engineering_failure"] for r in rows):
                                raise RuntimeError(
                                    f"{name} control has engineering failures"
                                )
                            state["controls"][key] = str(control.relative_to(directory))
                    entry["status"] = "completed"
                    entry.pop("error", None)
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    if getattr(exc, "run_dir", None):
                        entry["run_dir"] = str(exc.run_dir)
                _json(directory / "study.json", state)
                _report(directory, plan, state)
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            _json(directory / "study.json", state)
            raise
        state["status"] = (
            "completed"
            if all(e.get("status") == "completed" for e in state["entries"].values())
            else "failed"
        )
        _json(directory / "study.json", state)
        _report(directory, plan, state)
    return show_study(directory)


def _report(directory, plan, state):
    rows = []
    for spec in plan["entries"]:
        entry = state["entries"].get(spec["id"], {})
        if entry.get("evaluation_dir"):
            report = Path(entry["evaluation_dir"]) / "run.json"
            if report.exists():
                data = json.loads(report.read_text())
                rows.append(
                    {
                        "id": spec["id"],
                        "H": spec["H"],
                        "V": spec["V"],
                        "algorithm": spec["algorithm"],
                        "transport": spec["transport"],
                        "status": entry["status"],
                        "evaluation": data,
                    }
                )
    controls = []
    for spec in plan["entries"]:
        initial = directory / "controls" / (spec["id"] + "_initial.json")
        if initial.exists():
            controls.append(
                {
                    "id": spec["id"] + "_initial",
                    "H": spec["H"],
                    "V": spec["V"],
                    "algorithm": spec["algorithm"] + "_initial",
                    "transport": spec["transport"],
                    "status": (
                        "failed"
                        if any(
                            r["engineering_failure"]
                            for r in json.loads(initial.read_text())
                        )
                        else "completed"
                    ),
                    "evaluation": {"results": json.loads(initial.read_text())},
                }
            )
    for key, path in state["controls"].items():
        h_name, level, transport, kind = key.split("_")
        spec = next(
            e
            for e in plan["entries"]
            if e["H_case"] == h_name
            and e["V_case"] == level
            and e["transport"] == transport
        )
        controls.append(
            {
                "id": key,
                "H": spec["H"],
                "V": spec["V"],
                "algorithm": kind,
                "transport": transport,
                "status": "completed",
                "evaluation": {"results": json.loads((directory / path).read_text())},
            }
        )
    rows.extend(controls)
    for row in rows:
        spec = next(
            e
            for e in plan["entries"]
            if row["id"].startswith(e["H_case"] + "_" + e["V_case"] + "_")
        )
        values = [
            dataset["V"][spec["V_case"]]
            for label, dataset in plan["datasets"].items()
            if label.startswith("test_")
        ]
        row["V_train"] = row.pop("V")
        row["V_test"] = sum(values) / len(values)
        row["V_test_by_case"] = values
    _json(
        directory / "summary.json",
        {"development_pilot": True, "rows": rows, "state": state},
    )
    flattened = [
        {k: v for k, v in row.items() if k != "evaluation"}
        | summarize(row["evaluation"].get("results", []))
        for row in rows
    ]
    if flattened:
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
