"""Persistent sampled stage envelopes used by native and Tune admission.

This is a resource estimate, not scientific identity or operating-system isolation.
Unknown phases retain explicit margins. Workers reserve the full phase envelope:
phase transitions do not constitute new admission opportunities.
"""

import copy
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments.evidence import write_json
from smartsom.experiments.locking import acquire, release

SCHEMA = "smartsom.stage-resource-model/v1"
MEMORY_FLOOR = 256 * 1024**2
UNKNOWN_FACTOR = 1.5
CPU_PEAK_FACTOR = 1.1
MIN_CPU_WINDOW = 0.1
STAGES = {"startup", "sampling", "optimizing", "validation", "evaluation", "saving"}


def _scenario_shape(value):
    value.pop("seed", None)
    factory = value.get("factory", {})
    factory.pop("name", None)
    for collection in ("machines", "buffers", "agvs", "ports", "inspection_stations"):
        for entity in factory.get(collection, []):
            entity.pop("name", None)
    return value


def describe(
    prepared, *, task=None, numerical_threads=None, sampling_child_threads=None
):
    """Exclude only known run labels/seeds; retain physical and model semantics."""
    if (
        not prepared
        or not {
            "config_json",
            "scenario_json",
            "composition_json",
            "parameters_json",
            "policies_json",
        }
        <= prepared.keys()
    ):
        return None
    from smartsom.experiments.performance_profiles import policy_shape

    config = json.loads(prepared["config_json"])
    config = copy.deepcopy(config)
    if numerical_threads is not None:
        config.setdefault("runtime", {})["numerical_threads"] = numerical_threads
    layout = config.setdefault("runtime", {})
    layout["sampling_child_threads"] = (
        (
            sampling_child_threads
            if sampling_child_threads is not None
            else layout.get("numerical_threads", 1)
        )
        if layout.get("sampling_processes", 0)
        else 0
    )
    for key in (
        "seed",
        "data_seed",
        "name",
        "output",
        "logging",
        "scenario",
        "composition",
        "scenario_overrides",
    ):
        config.pop(key, None)
    for section in ("training", "validation", "evaluation"):
        if isinstance(config.get(section), dict):
            config[section].pop("seed", None)
            config[section].pop("parameters", None)
            # Complete resolved validation/evaluation cases are included below.
            config[section].pop("scenarios", None)
    training = config.get("training") or {}
    shape = {"contract": SCHEMA, "config": config}
    for field in (
        "scenario_json",
        "composition_json",
        "parameters_json",
        "training_inputs_json",
        "validation_json",
        "evaluation_json",
    ):
        if field in prepared:
            value = json.loads(prepared[field])
            # Only documented case-level RNG/authoring metadata is excluded.
            if isinstance(value, dict):
                value.pop("seed", None)
                value.pop("data_seed", None)
                value.pop("authoring", None)
                if isinstance(value.get("settings"), dict):
                    value["settings"].pop("factory", None)
                    value["settings"].pop("workload", None)
                if field == "scenario_json":
                    _scenario_shape(value)
                if field == "composition_json":
                    if isinstance(value.get("matching"), dict):
                        value.pop("pickup_matching", None)
                    for declaration in value.get("groups", {}).values():
                        declaration.pop("policy", None)
            elif isinstance(value, list):
                for case in value:
                    if isinstance(case, dict):
                        case.pop("seed", None)
                        case.pop("policy_seed", None)
                        if isinstance(case.get("scenario"), dict):
                            _scenario_shape(case["scenario"])
                        settings = case.get("recipe", {}).get("settings", {})
                        settings.pop("factory", None)
                        settings.pop("workload", None)
            shape[field] = value
    if "policies_json" in prepared:
        shape["policies"] = policy_shape(
            json.loads(prepared["policies_json"]),
            training_groups=training.get("groups", ()),
        )
    # Implementation changes invalidate reusable resource evidence, independently
    # of scientific checkpoint identity.
    from smartsom.experiments.composable import implementation_identity

    shape["implementation"] = implementation_identity()
    stages = ["startup", "saving"]
    if task != "evaluate" and training:
        stages += ["sampling", "optimizing"]
        if (config.get("validation") or {}).get("enabled", False):
            stages.append("validation")
    if task in {"evaluate", "train-evaluate"} or not training:
        stages.append("evaluation")
    return {"key": digest(shape), "stages": sorted(set(stages))}


def snapshot(root):
    path = Path(root) / "control/resource-model.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {"schema": SCHEMA, "groups": {}}
    if data.get("schema") != SCHEMA:
        raise ValueError("unsupported runtime resource model")
    return data


@contextmanager
def _transaction(root):
    directory = Path(root) / "control"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "resource-model.lock").open("a+") as stream:
        while True:
            try:
                acquire(stream)
                break
            except BlockingIOError:
                time.sleep(0.01)
        try:
            state = snapshot(root)
            yield state
            write_json(directory / "resource-model.json", state)
        finally:
            release(stream)


def observe(root, descriptor, stage, *, memory, cpus=None):
    """Record only actual samples; unavailable CPU is distinct from measured zero."""
    if descriptor is None or memory is None:
        return
    if stage not in STAGES:
        stage = "unknown"
    if (
        not math.isfinite(memory)
        or memory < 0
        or (cpus is not None and (not math.isfinite(cpus) or cpus < 0))
    ):
        raise ValueError("resource observations must be nonnegative finite values")
    with _transaction(root) as state:
        group = state["groups"].setdefault(descriptor["key"], {"stages": {}})
        row = group["stages"].setdefault(
            stage,
            {"memory_peak": 0, "cpu_peak": None, "memory_samples": 0, "cpu_samples": 0},
        )
        row["memory_peak"] = max(row["memory_peak"], int(memory))
        row["memory_samples"] += 1
        if cpus is not None:
            row["cpu_peak"] = max(row["cpu_peak"] or 0, cpus)
            row["cpu_samples"] += 1
        row["last_observed_at"] = time.time()
        group["expected_stages"] = descriptor["stages"]


def estimate(root, descriptor, *, cpus, memory):
    """Return raw request; the admission layer adds its usual RAM peak factor.

    Reading an atomically replaced model never acquires its write lock, so this
    also works inside admission.lock without introducing a nested lock cycle.
    """
    if descriptor is None:
        return {"cpus": cpus, "memory": memory}
    rows = snapshot(root)["groups"].get(descriptor["key"], {}).get("stages", {})
    unknown = [
        stage
        for stage in descriptor["stages"]
        if not rows.get(stage, {}).get("memory_samples")
        or not rows.get(stage, {}).get("cpu_samples")
    ]
    memory_base = max(MEMORY_FLOOR, memory)
    memory_peak = max([memory_base, *(row["memory_peak"] for row in rows.values())])
    cpu_peak = max(
        [cpus, *((row["cpu_peak"] or 0) * CPU_PEAK_FACTOR for row in rows.values())]
    )
    return {
        "cpus": cpu_peak,
        "memory": math.ceil(memory_peak * (UNKNOWN_FACTOR if unknown else 1.0)),
        "resource_group": descriptor["key"],
        "expected_stages": descriptor["stages"],
        "unknown_stages": unknown,
        "memory_floor": MEMORY_FLOOR,
        "unknown_factor": UNKNOWN_FACTOR,
        "cpu_peak_factor": CPU_PEAK_FACTOR,
        "input_cpus": cpus,
        "input_memory": memory,
    }


def stage_at(directory):
    """Use live worker telemetry; a missing/ambiguous phase stays unknown."""
    directory = Path(directory)
    paths = [
        directory / "logs/progress.json",
        *directory.glob("entries/*/logs/progress.json"),
    ]
    latest = None
    for path in paths:
        try:
            progress = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rows = progress.get("tasks", [])
        for row in rows:
            phase = row.get("stage")
            if phase in STAGES and row.get("status") not in {
                "completed",
                "failed",
                "stopped",
                "paused",
            }:
                sample = (row.get("updated_at", progress.get("updated_at", 0)), phase)
                if latest is None or sample[0] > latest[0]:
                    latest = sample
    return latest[1] if latest else "unknown"


class TreeSampler:
    """Sample process-tree CPU deltas; first samples and phase changes are unknown."""

    def __init__(self):
        self.previous = {}
        self.accumulated = {}
        self.identities = {}

    def sample(self, pid, stage):
        import psutil

        try:
            parent = psutil.Process(pid)
            processes = [parent, *parent.children(recursive=True)]
            now = time.monotonic()
            current = {}
            memory = 0
            for process in processes:
                identity = (process.pid, process.create_time())
                counters = process.cpu_times()
                current[identity] = counters.user + counters.system
                memory += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.identities.pop(pid, None)
            return None, None
        self.identities[pid] = list(current)
        prior = self.previous.get(pid)
        self.previous[pid] = (now, stage, current)
        cpu = None
        if (
            prior
            and prior[1] == stage
            and now > prior[0]
            and current.keys() == prior[2].keys()
        ):
            consumed = sum(
                max(0, value - prior[2][identity])
                for identity, value in current.items()
            )
            prior_cpu, prior_wall = self.accumulated.get((pid, stage), (0, 0))
            consumed += prior_cpu
            elapsed = now - prior[0] + prior_wall
            if elapsed >= MIN_CPU_WINDOW:
                cpu = consumed / elapsed
                self.accumulated[(pid, stage)] = (0, 0)
            else:
                self.accumulated[(pid, stage)] = (consumed, elapsed)
        return memory, cpu


_PHASE_RECORDERS = {}


def phase_boundary(directory, stage, *, prepared=None, sampling_child_threads=None):
    """Observe actual learner phase boundaries without touching learner state."""
    import os
    from uuid import uuid4

    if stage not in STAGES:
        stage = "startup" if stage == "initializing" else "unknown"
    key = (os.getpid(), str(directory), id(prepared))
    if key not in _PHASE_RECORDERS:
        from dataclasses import asdict

        descriptor = (
            describe(
                asdict(prepared),
                task="train-evaluate",
                sampling_child_threads=sampling_child_threads,
            )
            if prepared is not None
            else None
        )
        _PHASE_RECORDERS[key] = (
            TreeSampler(),
            {"id": uuid4().hex, "stages": {}, "resource_descriptor": descriptor},
            stage,
        )
    sampler, evidence, previous = _PHASE_RECORDERS[key]
    memory, cpus = sampler.sample(os.getpid(), previous)
    if memory is not None:
        row = evidence["stages"].setdefault(
            previous,
            {"memory_peak": 0, "cpu_peak": None, "memory_samples": 0, "cpu_samples": 0},
        )
        row["memory_peak"] = max(row["memory_peak"], memory)
        row["memory_samples"] += 1
        if cpus is not None:
            row["cpu_peak"] = max(row["cpu_peak"] or 0, cpus)
            row["cpu_samples"] += 1
    if stage != previous:
        sampler.sample(os.getpid(), stage)
    _PHASE_RECORDERS[key] = (sampler, evidence, stage)
    evidence["current_stage"] = stage
    evidence["updated_at"] = time.time()
    (Path(directory) / "control").mkdir(parents=True, exist_ok=True)
    write_json(Path(directory) / "control/stage-resources.json", evidence)


def ingest(root, descriptor, directory):
    """Merge worker boundary samples once per evidence generation/sample count."""
    if descriptor is None:
        return
    directory = Path(directory)
    paths = [
        directory / "control/stage-resources.json",
        *directory.glob("runs/*/control/stage-resources.json"),
        *directory.glob("entries/*/runs/*/control/stage-resources.json"),
    ]
    evidence = []
    for path in paths:
        try:
            evidence.append(json.loads(path.read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    if not evidence:
        return
    with _transaction(root) as state:
        for sample in evidence:
            actual = sample.get("resource_descriptor") or descriptor
            group = state["groups"].setdefault(actual["key"], {"stages": {}})
            group["expected_stages"] = actual["stages"]
            seen = group.setdefault("sources", {})
            previous = seen.setdefault(sample["id"], {})
            for stage, row in sample["stages"].items():
                saved = group["stages"].setdefault(
                    stage,
                    {
                        "memory_peak": 0,
                        "cpu_peak": None,
                        "memory_samples": 0,
                        "cpu_samples": 0,
                    },
                )
                saved["memory_peak"] = max(saved["memory_peak"], row["memory_peak"])
                if row["cpu_peak"] is not None:
                    saved["cpu_peak"] = max(saved["cpu_peak"] or 0, row["cpu_peak"])
                for metric in ("memory_samples", "cpu_samples"):
                    saved[metric] += max(
                        0, row[metric] - previous.get(stage, {}).get(metric, 0)
                    )
                saved["last_observed_at"] = sample["updated_at"]
                previous[stage] = {
                    k: max(row[k], previous.get(stage, {}).get(k, 0))
                    for k in ("memory_samples", "cpu_samples")
                }
