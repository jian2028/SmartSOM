"""Frozen batch inputs and durable, driver-owned Tune execution segments."""

import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import digest
from smartsom.config.experiment_v3 import load_v3, prepare_v3, read_document
from smartsom.experiments.evidence import (
    runtime_source_matches,
    source_identity,
    write_json,
)

SCHEMA = "smartsom.tune-batch/v1"


def _framework_call(operation, *args, **kwargs):
    """Keep Ray's human-readable diagnostics off structured CLI stdout."""
    import sys
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        return operation(*args, **kwargs)


@dataclass(frozen=True)
class BatchInputs:
    entries: tuple
    mode: str = "balanced"
    execution: str = "adaptive"
    active_limit: float = 600.0
    output_root: str = "runs"
    provenance: dict | None = None
    preflight: str = "quick"
    preflight_coverage: str = "each"
    calibration_level: str = "quick"
    calibration_candidate: str = "latest"

    def __post_init__(self):
        if self.mode not in {"balanced", "performance"} or self.execution not in {
            "adaptive",
            "fixed",
        }:
            raise ValueError(
                "mode must be balanced/performance and execution adaptive/fixed"
            )
        if (
            isinstance(self.active_limit, bool)
            or not math.isfinite(self.active_limit)
            or not 0 < self.active_limit
        ):
            raise ValueError("calibration timeout must be positive and finite")
        if self.preflight not in {"quick", "full"} or self.preflight_coverage not in {
            "each",
            "representative",
        }:
            raise ValueError("invalid preflight level or coverage")
        if (
            self.calibration_level not in {"quick", "full"}
            or not self.calibration_candidate
        ):
            raise ValueError("invalid calibration level or candidate")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value
    ):
        raise ValueError(
            "experiment id must be a safe unique name of at most 96 characters"
        )
    return value


def _contained(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("frozen input path escapes its study directory")
    return path


def load_batch(batch=None, study=None):
    """Read a manifest or import study *inputs*, never mutate/resume that study."""
    if (batch is None) == (study is None):
        raise ValueError("provide exactly one of --batch and --study")
    entries = []
    if batch is not None:
        path = Path(batch).expanduser().resolve()
        data = read_document(path)
        if data.get("schema") != SCHEMA or set(data) - {
            "schema",
            "entries",
            "mode",
            "execution",
            "active_limit",
            "output_root",
        }:
            raise ValueError("invalid Tune batch schema or unknown field")
        for row in data.get("entries", []):
            if set(row) != {"id", "config"}:
                raise ValueError("batch entries require exactly id and config")
            prepared = prepare_v3(
                load_v3(path.parent / row["config"]),
                training=True,
                require_dependencies=True,
            )
            entries.append(
                {
                    "experiment_id": _identifier(row["id"]),
                    "prepared": asdict(prepared),
                    "control_spec": {},
                }
            )
        options = {
            key: data[key]
            for key in ("mode", "execution", "active_limit")
            if key in data
        }
        output = str((path.parent / data.get("output_root", "runs")).resolve())
        provenance = {"kind": "batch", "path": str(path), "sha256": digest(data)}
    else:
        from smartsom.experiments.composable import prepared_from_run

        root = Path(study).expanduser().resolve()
        plan = json.loads((root / "plan.json").read_text())
        state = json.loads((root / "study.json").read_text())
        if plan.get("schema") != "smartsom.composable-study-plan/v1" or digest(
            plan
        ) != state.get("plan_sha256"):
            raise ValueError("frozen study plan changed or is unsupported")
        for row in plan["entries"]:
            snapshot = _contained(root, row["snapshot"])
            if digest(json.loads(snapshot.read_text())) != row["snapshot_sha256"]:
                raise ValueError("frozen study child snapshot changed")
            prepared = prepared_from_run(snapshot.parent.parent)
            if prepared.scientific_sha256 != row["scientific_sha256"]:
                raise ValueError("study scientific identity changed")
            pair_key = digest(
                {
                    "scenario": prepared.scenario_json,
                    "evaluation": prepared.evaluation_json,
                }
            )
            entries.append(
                {
                    "experiment_id": _identifier(row["id"]),
                    "prepared": asdict(prepared),
                    "control_spec": {
                        "names": plan["recipe"].get(
                            "controls", ["initial", "rule", "random"]
                        ),
                        "pair_key": pair_key,
                    },
                }
            )
        options, output = {}, str(root.parent / "tune-runs")
        provenance = {
            "kind": "study-input-import",
            "directory": str(root),
            "plan_sha256": digest(plan),
            "original_source": plan["source"],
        }
    if not entries:
        raise ValueError("a batch needs at least one training experiment")
    identities = [row["experiment_id"] for row in entries]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate experiment id")
    result = BatchInputs(
        tuple(entries), output_root=output, provenance=provenance, **options
    )
    if result.mode not in {"balanced", "performance"} or result.execution not in {
        "adaptive",
        "fixed",
    }:
        raise ValueError(
            "mode must be balanced/performance; execution must be adaptive/fixed"
        )
    if isinstance(result.active_limit, bool) or not 0 < result.active_limit:
        raise ValueError("calibration timeout must be positive")
    return result


def preflight(inputs):
    """No learner, Ray allocation, calibration or training is started here."""
    from smartsom.config.experiment_v3 import PreparedComposition

    for module in ("ray", "torch", "psutil", "threadpoolctl"):
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f"Tune requires optional dependency {module}; install the tuning and matching learning extras"
            )
    import ray

    if ray.__version__ != "2.58.0":
        raise ValueError("Tune adapter requires locked Ray 2.58.0")
    rows = []
    for entry in inputs.entries:
        prepared = PreparedComposition(**entry["prepared"])
        config = prepared.config
        if config.training is None:
            raise ValueError("Tune supports v3 training experiments")
        module = "sb3_contrib" if config.training.backend == "sb3" else "ray.rllib"
        if importlib.util.find_spec(module) is None:
            raise ImportError(f"missing learning backend: {module}")
        if config.runtime.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("frozen CUDA experiment requires available CUDA")
        # Validate every frozen world/recipe, without regenerating random worlds.
        prepared.scenario
        json.loads(prepared.validation_json)
        json.loads(prepared.evaluation_json)
        if config.runtime.sampling_processes and config.runtime.num_envs > 1:
            issue = _parallel_sampling_issue(prepared)
            if issue:
                raise ValueError(issue)
        rows.append(
            {
                "experiment_id": entry["experiment_id"],
                "algorithm": config.training.algorithm,
                "backend": config.training.backend,
                "device": config.runtime.device,
                "starting_layout": {
                    "num_envs": config.runtime.num_envs,
                    "sampling_processes": config.runtime.sampling_processes,
                    "threads": config.runtime.numerical_threads,
                    "concurrency": entry.get("baseline_concurrency", 1),
                },
                "physical_ticks": config.training.total_ticks,
            }
        )
    parent = Path(inputs.output_root)
    existing = parent
    while not existing.exists():
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK):
        raise ValueError("batch output root is not writable")
    return {
        "status": "feasible",
        "experiments": rows,
        "mode": inputs.mode,
        "execution": inputs.execution,
        "active_limit": inputs.active_limit,
        "output_root": inputs.output_root,
        "calibration_level": inputs.calibration_level,
        "calibration_candidate": inputs.calibration_candidate,
        "resource_wait": "included in total calibration wall-clock timeout",
    }


# Keep the public batch check name while allowing run_batch's preflight option.
check_batch_inputs = preflight


def _parallel_sampling_issue(prepared):
    from smartsom.experiments.composable import (
        parallel_sampling_issue,
        policies_for,
        verify_prepared_rules,
    )

    verify_prepared_rules(prepared)
    policies, _ = policies_for(prepared, training=False)
    return parallel_sampling_issue(policies)


def allocate_batch(inputs):
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.experiments.composable import archive_inputs, implementation_identity

    root = Path(inputs.output_root) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-tune-" + uuid4().hex[:10]
    )
    root.mkdir(parents=True)
    source = source_identity()
    frozen = []
    for entry in inputs.entries:
        item = dict(entry)
        target = root / "inputs" / item["experiment_id"]
        (target / "config").mkdir(parents=True)
        item["prepared"] = asdict(
            archive_inputs(target, PreparedComposition(**item["prepared"]))
        )
        item["control_spec"] = {
            **item["control_spec"],
            "directory": str(root / "controls"),
        }
        frozen.append(item)
    plan = {
        "schema": SCHEMA,
        "entries": frozen,
        "mode": inputs.mode,
        "execution": inputs.execution,
        "active_limit": inputs.active_limit,
        "output_root": inputs.output_root,
        "calibration_level": inputs.calibration_level,
        "calibration_candidate": inputs.calibration_candidate,
        "preflight": inputs.preflight,
        "preflight_coverage": inputs.preflight_coverage,
        "source": source,
        "implementation_sha256": implementation_identity(),
        "provenance": inputs.provenance,
    }
    write_json(root / "plan.json", plan)
    ledger = {
        "schema": "smartsom.tune-state/v1",
        "plan_sha256": digest(plan),
        "status": "prepared",
        "entries": {e["experiment_id"]: {"status": "queued"} for e in frozen},
        "segments": [],
    }
    write_json(root / "batch.json", ledger)
    write_json(
        root / "run.json",
        {
            "schema": "smartsom.tune-run/v1",
            "kind": "tune",
            "status": "prepared",
            "stage": "preflight",
            "name": root.name,
        },
    )
    return root, plan, ledger


def load_run(root):
    from smartsom.experiments.composable import implementation_identity

    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if plan.get("schema") != SCHEMA or digest(plan) != state.get("plan_sha256"):
        raise ValueError("frozen Tune plan changed")
    live = source_identity()
    author_batch = (plan.get("provenance") or {}).get("kind") == "author-batch"
    if (
        not (
            runtime_source_matches(plan["source"], live)
            if author_batch
            else live == plan["source"]
        )
        or implementation_identity() != plan["implementation_sha256"]
    ):
        raise ValueError(
            "batch source/dependency identity changed; import inputs into a new batch"
        )
    from smartsom.experiments.composable import prepared_from_run

    for entry in plan["entries"]:
        checked = prepared_from_run(root / "inputs" / entry["experiment_id"])
        if asdict(checked) != entry["prepared"]:
            raise ValueError("frozen batch input or partner model changed")
    recover_committed(root, plan, state)
    return root, plan, state


def _groups(entries):
    """Profile equivalent learner shapes; select the most demanding frozen world."""
    from smartsom.config.experiment_v3 import PreparedComposition

    result, mapping = {}, {}
    for entry in entries:
        prepared = PreparedComposition(**entry["prepared"])
        config = prepared.config
        policies = json.loads(prepared.policies_json)
        matching = json.loads(prepared.composition_json).get("matching")
        factory = json.loads(prepared.scenario_json)["factory"]
        key = digest(
            {
                "backend": config.training.backend,
                "algorithm": config.training.algorithm,
                "training_groups": config.training.groups,
                "training_mode": config.training.mode,
                "gamma": config.training.gamma,
                "reward": config.training.reward.model_dump(mode="json"),
                "pickup_matching": matching,
                "update_quantum": min(
                    config.training.total_ticks, config.training.ticks_per_update
                ),
                "episode_tick_limit": config.training.max_ticks,
                "factory": factory,
                "parameters": prepared.parameters_json,
                "policies": {
                    g: {
                        k: d.get(k)
                        for k in ("role", "network", "projection", "implementation")
                    }
                    for g, d in policies.items()
                },
                "device": config.runtime.device,
            }
        )[:16]
        mapping[entry["experiment_id"]] = key
        workload = json.loads(prepared.training_inputs_json)
        score = (
            len(prepared.scenario_json)
            + len(prepared.validation_json)
            + len(json.dumps(workload))
        )
        if key not in result or score > result[key]["representative_score"]:
            result[key] = {
                "prepared": entry["prepared"],
                "representative_score": score,
                "group": key,
                "cpu_overhead": 0,
                "device": config.runtime.device,
                "baseline": {
                    "threads": config.runtime.numerical_threads,
                    "num_envs": config.runtime.num_envs,
                    "sampling_processes": config.runtime.sampling_processes,
                    "concurrency": entry.get("baseline_concurrency", 1),
                },
                "parallel_sampling_issue": _parallel_sampling_issue(prepared),
            }
    return result, mapping


def _short_probe_prepared(encoded, *, tick_limit=512):
    """Derive disposable one-case calibration input from a frozen V4 entry."""
    from dataclasses import replace

    from smartsom.config.codec import canonical_json
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.config.experiment_v4 import scientific_identity

    original = PreparedComposition(**encoded)
    if original.config.schema_id != "smartsom.execution-config/v2":
        return encoded
    validation = json.loads(original.validation_json)
    if not validation:
        raise ValueError("batch calibration requires a frozen validation case")
    config = json.loads(original.config_json)
    config["training"]["total_ticks"] = min(
        config["training"]["total_ticks"], tick_limit
    )
    config["training"]["max_ticks"] = min(config["training"]["max_ticks"], tick_limit)
    config["training"]["ticks_per_update"] = min(
        config["training"]["ticks_per_update"], tick_limit
    )
    scenario = json.loads(original.scenario_json)
    scenario["tick_limit"] = tick_limit
    training_inputs = json.loads(original.training_inputs_json)
    training_inputs["settings"]["tick_limit"] = tick_limit
    case = validation[0]
    case["scenario"]["tick_limit"] = tick_limit
    case["recipe"]["settings"]["tick_limit"] = tick_limit
    probe = replace(
        original,
        config_json=canonical_json(config),
        scenario_json=canonical_json(scenario),
        training_inputs_json=canonical_json(training_inputs),
        validation_json=canonical_json([case]),
    )
    return asdict(replace(probe, scientific_sha256=scientific_identity(probe)))


def _algorithm_display_name(prepared, identity):
    authoring = json.loads(prepared.training_inputs_json).get("authoring", {})
    source = authoring.get("sources", {}).get("algorithm")
    if source:
        return f"{Path(source).stem} · seed {prepared.config.seed} ({identity})"
    return identity


def _with_selected_layout(prepared, profile):
    """Freeze the measured sampler layout into a new pre-training identity."""
    from dataclasses import replace

    config = json.loads(prepared.config_json)
    config["runtime"].update(
        num_envs=profile["num_envs"],
        sampling_processes=profile["sampling_processes"],
        numerical_threads=profile["threads"],
    )
    selected = replace(prepared, config_json=json.dumps(config, sort_keys=True))
    if config["schema"].startswith("smartsom.execution-config/"):
        from smartsom.config.experiment_v4 import scientific_identity

        return replace(selected, scientific_sha256=scientific_identity(selected))
    scientific_config = {
        **config,
        "logging": {
            key: value
            for key, value in config["logging"].items()
            if key not in {"title", "task_title"}
        },
    }
    identity = digest(
        {
            "config": scientific_config,
            "scenario": json.loads(prepared.scenario_json),
            "composition": json.loads(prepared.composition_json),
            "policies": json.loads(prepared.policies_json),
            "parameters": json.loads(prepared.parameters_json),
            "validation": json.loads(prepared.validation_json),
            "evaluation": json.loads(prepared.evaluation_json),
            "training_inputs": json.loads(prepared.training_inputs_json),
        }
    )
    return replace(selected, scientific_sha256=identity)


class CalibrationMonitor:
    """Exclude own CPU load without adding unleased driver/probe RAM back."""

    def __init__(self, monitor):
        self.monitor = monitor

    def snapshot(self, exclude_pids=()):
        from dataclasses import replace

        observed = self.monitor.snapshot(exclude_pids=exclude_pids)
        return replace(
            observed,
            processes=tuple(replace(p, excluded=False) for p in observed.processes),
        )


def _projection_compatible(entries, group):
    """A short sample can predict only matching update and case tick limits."""
    from smartsom.config.experiment_v3 import PreparedComposition

    probe = PreparedComposition(**group["prepared"])
    probe_cases = json.loads(probe.validation_json)
    if not probe_cases:
        return False
    probe_limit = probe_cases[0]["scenario"]["tick_limit"]
    for entry in entries:
        prepared = PreparedComposition(**entry["prepared"])
        config = prepared.config
        if (
            config.training.ticks_per_update != probe.config.training.ticks_per_update
            or config.training.total_ticks % config.training.ticks_per_update
        ):
            return False
        if any(
            case["scenario"]["tick_limit"] != probe_limit
            for case in json.loads(prepared.validation_json)
        ):
            return False
    return True


def _project_group_training(entries, group, stages, elapsed):
    """Estimate each full learner from one disposable update's measured phases.

    Startup/finish happens once per entry. Sampling, optimization and saving
    recur each update. Validation recurs at the frozen interval; its one-case
    short probe is scaled only by case count when case tick limits match.
    """
    from smartsom.config.experiment_v3 import PreparedComposition

    if not _projection_compatible(entries, group):
        return None
    update = sum(stages.get(name, 0.0) for name in ("sampling", "optimizing", "saving"))
    validation = stages.get("validation", 0.0)
    probe_cases = json.loads(PreparedComposition(**group["prepared"]).validation_json)
    if update <= 0 or elapsed <= 0 or (probe_cases and validation <= 0):
        return None
    one_time = max(0.0, elapsed - update - validation)
    probe_ticks = sum(c["scenario"]["tick_limit"] for c in probe_cases)
    if probe_cases and probe_ticks <= 0:
        return None
    estimate = 0.0
    for entry in entries:
        prepared = PreparedComposition(**entry["prepared"])
        config = prepared.config
        updates = math_ceil_div(
            config.training.total_ticks, config.training.ticks_per_update
        )
        validation_rounds = (
            updates // config.validation.every_updates
            if config.validation.enabled
            else 0
        )
        full_cases = json.loads(prepared.validation_json)
        validation_factor = len(full_cases) / len(probe_cases) if probe_cases else 0.0
        estimate += (
            one_time
            + updates * update
            + validation_rounds * validation * validation_factor
        )
    return estimate


def _batch_schedule(
    root,
    plan,
    groups,
    mapping,
    recommendations,
    measurements,
    supervisor,
    monitor,
    started,
    *,
    cancelled,
):
    """Compare two compatible groups by estimated training-stage completion time.

    No held-out evaluation is measured here; that cost is explicitly excluded.
    Without a valid mixed pilot, separate group waves are the safe schedule.
    """
    from smartsom.experiments.tuning_calibration import ExecutionProfile
    from smartsom.experiments.tuning_probe import run_mixed_training_probe
    from smartsom.experiments.tuning_resources import ResourceBroker, ResourceRequest

    ids = {group: [] for group in groups}
    grouped_entries = {group: [] for group in groups}
    for entry in plan["entries"]:
        group = mapping[entry["experiment_id"]]
        ids[group].append(entry["experiment_id"])
        grouped_entries[group].append(entry)
    schedule = {
        "status": "schedule_uncalibrated",
        "basis": "training-stage probe only; final held-out evaluation not measured",
        "waves": [members for members in ids.values() if members],
        "mixed_probe": None,
    }
    by_group = {}
    for measurement in measurements:
        if measurement.eligible and measurement.profile == recommendations.get(
            measurement.group
        ):
            by_group.setdefault(measurement.group, []).append(measurement)
    eligible = [
        group
        for group in groups
        if group in by_group and recommendations[group].device == "cpu"
    ]
    if len(eligible) < 2 or cancelled():
        return schedule
    # Prefer the largest two measured training costs. Each mixed worker gets
    # the corresponding real frozen representative, never a synthetic job.
    separate = {}
    for group in eligible:
        estimates = [
            _project_group_training(
                grouped_entries[group],
                groups[group],
                m.stages,
                m.stages.get("worker_0_seconds", 0),
            )
            for m in by_group[group]
        ]
        valid = sorted(value for value in estimates if value is not None)
        if valid:
            separate[group] = valid[len(valid) // 2]
    eligible = [group for group in eligible if group in separate]
    if len(eligible) < 2:
        schedule["reason"] = "baseline probes lack phase timing for training projection"
        return schedule
    eligible.sort(key=lambda group: -separate[group])
    a, b = eligible[:2]
    pa, pb = recommendations[a], recommendations[b]
    if pa.concurrency != 1 or pb.concurrency != 1:
        schedule["reason"] = (
            "mixed two-worker pilot cannot represent selected multi-worker groups"
        )
        return schedule
    if (pa.threads, pa.num_envs, pa.sampling_processes) != (
        pb.threads,
        pb.num_envs,
        pb.sampling_processes,
    ):
        schedule["reason"] = "selected groups require different worker layouts"
        return schedule
    best_a = max(by_group[a], key=lambda m: m.peak_memory)
    best_b = max(by_group[b], key=lambda m: m.peak_memory)
    memory = int(best_a.stages.get("per_trial_peak_memory", best_a.peak_memory)) + int(
        best_b.stages.get("per_trial_peak_memory", best_b.peak_memory)
    )
    request = ResourceRequest(2 * (pa.threads + pa.sampling_processes), memory, None, 0)
    observed = monitor.snapshot(exclude_pids=(os.getpid(),))
    if not ResourceBroker(mode=plan["mode"]).admit(request, observed).allowed:
        schedule["reason"] = "mixed workers exceed observed resource budget"
        return schedule
    remaining = plan["active_limit"] - (time.monotonic() - started)
    needed = max(best_a.elapsed_seconds, best_b.elapsed_seconds) * 1.2 + 1.0
    if remaining <= needed:
        schedule["reason"] = "insufficient calibration time for mixed pilot and cleanup"
        return schedule
    profile = ExecutionProfile(pa.threads, 2, "cpu", pa.num_envs, pa.sampling_processes)
    mixed = supervisor.run(
        run_mixed_training_probe,
        {
            "prepared": groups[a]["prepared"],
            "mixed_workers": [groups[a], groups[b]],
            "cpu_overhead": pa.sampling_processes,
        },
        profile,
        min(remaining, needed * 1.5),
        cancelled,
    )
    schedule["mixed_probe"] = asdict(mixed)
    if not mixed.eligible:
        schedule["reason"] = mixed.reason or "mixed pilot invalid"
        return schedule
    mixed_estimates = []
    for index, name in enumerate((a, b)):
        prefix = f"worker_{index}_"
        stages = {
            key.removeprefix(prefix): value
            for key, value in mixed.stages.items()
            if key.startswith(prefix)
        }
        mixed_estimates.append(
            _project_group_training(
                grouped_entries[name],
                groups[name],
                stages,
                mixed.stages.get(f"worker_{index}_seconds", 0),
            )
        )
    if any(value is None for value in mixed_estimates):
        schedule["reason"] = "mixed pilot lacks per-worker phase timing"
        return schedule
    serial_seconds = separate[a] + separate[b]
    mixed_seconds = max(mixed_estimates)
    schedule["estimated_training_seconds"] = {
        "separate": serial_seconds,
        "mixed": mixed_seconds,
    }
    if mixed_seconds < serial_seconds * 0.95:
        schedule["status"] = "mixed_measured"
        schedule["waves"] = [ids[a] + ids[b]] + [
            members for group, members in ids.items() if group not in {a, b} and members
        ]
        schedule["reason"] = (
            "mixed pilot predicts at least 5% shorter training makespan"
        )
    else:
        schedule["status"] = "separate_measured"
        schedule["incompatible_pairs"] = [[a, b]]
        schedule["reason"] = (
            "separate group waves predict shorter or equivalent training makespan"
        )
    return schedule


def calibrate(root, plan, *, display=None, monitor=None, supervisor=None):
    from smartsom.experiments.control import boundary, requested

    started_calibration = time.monotonic()
    boundary(root)
    from smartsom.experiments.performance_profiles import (
        group_shape,
        hardware_shape,
    )
    from smartsom.experiments.performance_profiles import (
        select as select_profile,
    )
    from smartsom.experiments.performance_profiles import (
        store as store_profiles,
    )
    from smartsom.experiments.tuning_calibration import (
        CalibrationController,
        ExecutionProfile,
        generate_candidates,
    )
    from smartsom.experiments.tuning_probe import ProbeSupervisor, run_training_probe
    from smartsom.experiments.tuning_resources import ResourceBroker, ResourceMonitor

    author_kind = (plan.get("provenance") or {}).get("kind")
    directory_batch = author_kind == "author-batch"
    author_v4 = author_kind in {"author-batch", "author-plan"}

    monitor = CalibrationMonitor(monitor or ResourceMonitor())
    groups, mapping = _groups(plan["entries"])
    snapshot = monitor.snapshot(exclude_pids=(os.getpid(),))
    observation_wait = 0.0
    if "CPU load first sample unavailable" in snapshot.unavailable:
        started = time.monotonic()
        time.sleep(0.25)
        snapshot = monitor.snapshot(exclude_pids=(os.getpid(),))
        observation_wait = time.monotonic() - started
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    # nvidia-smi physical indices are unambiguous only without a Slurm allocation;
    # the monitor refuses unresolved allocation GPU identifiers itself.
    tokens = (
        visible.split(",")
        if visible is not None
        else [gpu.index for gpu in snapshot.gpus]
    )
    capacity = ResourceBroker(mode=plan["mode"]).capacity(snapshot)
    level = plan.get("calibration_level", "quick")
    hardware = hardware_shape(snapshot, mode=plan["mode"])
    candidate_source = plan.get("calibration_candidate", "latest")
    state = {
        "active_seconds": 0.0,
        "waiting_seconds": observation_wait,
        "wall_seconds": time.monotonic() - started_calibration,
        "measured": 0,
        "limit_seconds": plan["active_limit"],
        "level": level,
        "candidate_source": candidate_source,
    }
    candidates, baselines, originals, profile_shapes, historical = {}, {}, {}, {}, {}
    for key, group in groups.items():
        if author_v4:
            group["source_prepared"] = group["prepared"]
            group["source_scientific_sha256"] = group["prepared"]["scientific_sha256"]
            tick_limit = (
                512
                if level == "quick"
                else min(
                    4096,
                    json.loads(group["prepared"]["validation_json"])[0]["scenario"][
                        "tick_limit"
                    ],
                )
            )
            group["prepared"] = _short_probe_prepared(
                group["prepared"], tick_limit=tick_limit
            )
            group["probe_scope"] = {
                "formal_evidence": False,
                "source": "frozen V4 input",
                "tick_limit": tick_limit,
                "validation_cases": 1,
            }
        group["visible_gpus"] = tokens
        device = group["device"]
        max_jobs = sum(value == key for value in mapping.values())
        initial = group["baseline"]
        baselines[key] = ExecutionProfile(
            initial["threads"],
            min(initial["concurrency"], max_jobs),
            "cpu" if device == "cpu" else "cuda:0",
            initial["num_envs"],
            initial["sampling_processes"],
        )
        originals[key] = baselines[key]
        if author_v4:
            profile_shapes[key] = group_shape(group, level=level)
            cached = select_profile(
                plan.get("output_root", root),
                hardware=hardware,
                shape=profile_shapes[key],
                candidate=candidate_source,
            )
        else:
            cached = None
        if cached is not None:
            chosen = ExecutionProfile(**cached["profile"])
            if (
                chosen.concurrency <= max_jobs
                and (chosen.threads + chosen.sampling_processes) * chosen.concurrency
                <= capacity.cpus
                and ((chosen.device == "cpu") == (device == "cpu"))
            ):
                baselines[key] = chosen
                historical[key] = cached
        candidates[key] = tuple(
            p
            for p in generate_candidates(snapshot, mode=plan["mode"])
            if (p.device == "cpu") == (device == "cpu")
            and p.concurrency <= max_jobs
            and (p.threads + p.sampling_processes) * p.concurrency <= capacity.cpus
            and (device == "cpu" or p.concurrency == 1)
            and (not group["parallel_sampling_issue"] or not p.sampling_processes)
        )

    def publish(stage, event):
        event = {
            key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
            for key, value in event.items()
        }
        if "waiting_seconds" in event:
            event["waiting_seconds"] += observation_wait
        if "wall_seconds" in event:
            event["wall_seconds"] = time.monotonic() - started_calibration
            event["remaining_seconds"] = max(
                0.0, plan["active_limit"] - event["wall_seconds"]
            )
        state.update(
            {
                key: value
                for key, value in event.items()
                if key
                in {
                    "active_seconds",
                    "wall_seconds",
                    "remaining_seconds",
                    "waiting_seconds",
                    "measured",
                    "candidates",
                    "reason",
                    "calibrated",
                    "profile",
                    "group",
                    "phase",
                }
            }
        )
        if display:
            observed = monitor.snapshot(exclude_pids=(os.getpid(),))
            current_capacity = ResourceBroker(mode=plan["mode"]).capacity(observed)
            display.configure_tuning(
                {
                    "stage": stage,
                    "calibration": dict(state),
                    "resources": {
                        "cpus_available": current_capacity.cpus,
                        "memory_available": current_capacity.memory,
                        "external_cpu_load": observed.external_cpu_load,
                        "mode": plan["mode"],
                    },
                    "entries": [
                        {"experiment_id": e["experiment_id"], "status": "queued"}
                        for e in plan["entries"]
                    ],
                }
            )

    def on_measure(event):
        publish("calibrating", event)

    def on_poll(event):
        current = dict(state)
        current["active_seconds"] = state["active_seconds"] + event["elapsed_seconds"]
        current["wall_seconds"] = time.monotonic() - started_calibration
        current["remaining_seconds"] = max(
            0.0, plan["active_limit"] - current["wall_seconds"]
        )
        current["phase"] = event.get("phase", "starting")
        if display:
            observed = monitor.snapshot(exclude_pids=(os.getpid(),))
            current_capacity = ResourceBroker(mode=plan["mode"]).capacity(observed)
            display.configure_tuning(
                {
                    "stage": "calibrating",
                    "calibration": current,
                    "resources": {
                        "cpus_available": current_capacity.cpus,
                        "memory_available": current_capacity.memory,
                        "external_cpu_load": observed.external_cpu_load,
                        "mode": plan["mode"],
                    },
                    "entries": [
                        {"experiment_id": e["experiment_id"], "status": "queued"}
                        for e in plan["entries"]
                    ],
                }
            )

    supervisor = supervisor or ProbeSupervisor(
        work_root=root / "calibration/probes", keep_artifacts=True, on_poll=on_poll
    )
    publish("calibrating", {"candidates": sum(len(c) + 1 for c in candidates.values())})
    # Leave time inside the one shared wall-clock deadline for the optional
    # mixed-worker pilot; otherwise a successful candidate search consumes it.
    mixable_groups = (
        [
            name
            for name, group in groups.items()
            if group["device"] == "cpu"
            and _projection_compatible(
                [
                    entry
                    for entry in plan["entries"]
                    if mapping[entry["experiment_id"]] == name
                ],
                group,
            )
        ]
        if directory_batch
        else []
    )
    mixed_reserve = (
        min(300.0, plan["active_limit"] * 0.2)
        if level == "full" and len(mixable_groups) >= 2
        else 0.0
    )
    report = CalibrationController(
        run_training_probe,
        monitor,
        mode=plan["mode"],
        active_limit=max(
            0.001,
            plan["active_limit"]
            - (time.monotonic() - started_calibration)
            - mixed_reserve,
        ),
        supervisor=supervisor,
        fair_baselines=directory_batch,
    ).run(
        groups,
        candidates,
        cancelled=lambda: requested(root),
        baseline_profiles=baselines,
        on_wait=lambda event: publish("waiting_resources", event),
        on_measure=on_measure,
    )
    fallback_groups = [name for name in groups if name not in report.recommendations]
    observed_profiles = {
        (measurement.group, measurement.profile) for measurement in report.measurements
    }
    untested = [
        {"group": name, "profile": asdict(profile)}
        for name, profiles in candidates.items()
        for profile in profiles
        if (name, profile) not in observed_profiles
    ]
    recommendations = {
        key: asdict(value) for key, value in report.recommendations.items()
    }
    if report.status != "cancelled":
        recommendations.update(
            {name: asdict(originals[name]) for name in fallback_groups}
        )
    baseline_errors = [
        m
        for m in report.measurements
        if m.group in baselines
        and m.profile == baselines[m.group]
        and not m.eligible
        and m.termination is None
    ]
    schedule = None
    if directory_batch and not baseline_errors:
        schedule = _batch_schedule(
            root,
            plan,
            groups,
            mapping,
            report.recommendations,
            report.measurements,
            supervisor,
            monitor,
            started_calibration,
            cancelled=lambda: requested(root),
        )
    payload = {
        "schema": "smartsom.tune-calibration/v1",
        "active_limit": plan["active_limit"],
        "mixed_probe_reserve_seconds": mixed_reserve,
        "active_seconds": report.active_seconds,
        "wall_seconds": time.monotonic() - started_calibration,
        "waiting_seconds": report.waiting_seconds + observation_wait,
        "measurements": [asdict(m) for m in report.measurements],
        "untested_candidates": untested,
        "tested_count": len(observed_profiles),
        "skipped_count": sum(
            m.reason is not None and "skipped" in m.reason for m in report.measurements
        ),
        "recommendations": recommendations,
        "groups": mapping,
        "status": report.status,
        "reason": report.reason,
        "missing_groups": list(report.missing_groups),
        "converged_groups": list(report.converged_groups),
        "unstable_groups": list(report.unstable_groups),
        "ready": report.status != "cancelled" and bool(recommendations),
        "uncalibrated_groups": fallback_groups,
        "calibrated": not fallback_groups,
        "calibration_level": level,
        "historical_candidates": historical,
        "hardware_shape": hardware,
        "profile_shapes": profile_shapes,
        "source": plan.get("source", {}),
        "probe_version": "v4-disposable-update-validation/2",
    }
    if schedule is not None:
        payload["schedule"] = schedule
    history = root / "calibration/reports"
    history.mkdir(parents=True, exist_ok=True)
    payload["report_file"] = str((history / (uuid4().hex + ".json")).relative_to(root))
    records = []
    for measurement in report.measurements:
        if author_v4 and measurement.eligible:
            records.append(
                {
                    "at": time.time(),
                    "hardware": hardware,
                    "shape": profile_shapes[measurement.group],
                    "profile": asdict(measurement.profile),
                    "throughput": measurement.throughput,
                    "source": plan["source"],
                    "report": str(root / payload["report_file"]),
                }
            )
    payload["profile_records"] = records
    write_json(root / payload["report_file"], payload)
    write_json(root / "calibration.json", payload)
    if author_v4:
        store_profiles(plan.get("output_root", root), records)
    if directory_batch:
        if baseline_errors:
            raise RuntimeError(
                "baseline calibration engineering error: "
                + "; ".join(f"{m.group}: {m.reason}" for m in baseline_errors)
            )
    boundary(root)
    publish(
        "calibration_complete" if report.ready else "calibration_incomplete",
        {
            "active_seconds": report.active_seconds,
            "waiting_seconds": report.waiting_seconds,
            "reason": report.reason,
            "calibrated": payload["calibrated"],
        },
    )
    if display:
        display.configure_tuning(
            {
                "stage": "calibration_complete"
                if report.ready
                else "calibration_incomplete",
                "calibration": {
                    **state,
                    "recommendation": payload["recommendations"],
                    "calibrated": payload["calibrated"],
                },
            }
        )
    if not payload["ready"] and not (directory_batch and report.status != "cancelled"):
        raise RuntimeError(
            "no valid constrained calibration baseline for: "
            + ", ".join(report.missing_groups)
        )
    return payload


def _execute_batch(
    root, plan, state, *, recommend_only=False, display=None, retry_failed=False
):
    """A resumed batch creates new segments from verified native committed state."""
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.experiments.tuning_resources import ResourceMonitor
    from smartsom.experiments.tuning_session import verify_identity

    root = Path(root)
    if display:
        display.bind(root)
        display.total_tasks = len(plan["entries"])
        from smartsom.telemetry.workflow import describe_prepared

        with display.batch_updates():
            for entry in plan["entries"]:
                previous = state["entries"][entry["experiment_id"]]
                prepared = PreparedComposition(**entry["prepared"])
                display.update(
                    entry["experiment_id"],
                    {
                        "status": previous["status"],
                        "stage": previous["status"],
                        "physical_ticks": previous.get("physical_ticks", 0),
                        "workflow": describe_prepared(prepared, "train-evaluate"),
                        "display_name": _algorithm_display_name(
                            prepared, entry["experiment_id"]
                        ),
                    },
                    total=prepared.config.training.total_ticks,
                    unit="physical ticks",
                    final=previous["status"] in {"completed", "failed"},
                )
    try:
        if all(e["status"] == "completed" for e in state["entries"].values()):
            state["status"] = "completed"
            return {
                "directory": str(root),
                "status": "completed",
                "completed": len(state["entries"]),
                "failed": 0,
            }
        eligible = [
            entry
            for entry in plan["entries"]
            if state["entries"][entry["experiment_id"]]["status"] != "completed"
            and (
                retry_failed
                or state["entries"][entry["experiment_id"]]["status"] != "failed"
            )
        ]
        if not eligible:
            state["status"] = "failed"
            return {
                "directory": str(root),
                "status": "failed",
                "completed": sum(
                    e["status"] == "completed" for e in state["entries"].values()
                ),
                "failed": sum(
                    e["status"] == "failed" for e in state["entries"].values()
                ),
            }
        state["status"] = "calibrating"
        write_json(root / "batch.json", state)
        saved_calibration = root / "calibration.json"
        saved = (
            json.loads(saved_calibration.read_text())
            if saved_calibration.exists()
            else {}
        )
        author_batch = (plan.get("provenance") or {}).get("kind") == "author-batch"
        frozen_v4 = (plan.get("provenance") or {}).get("kind") in {
            "author-batch",
            "author-plan",
        }
        current_groups = _groups(eligible)[1]
        reusable = (
            frozen_v4
            and saved.get("ready")
            and saved.get("frozen_plan_sha256") == digest(plan)
            and set(current_groups) <= set(saved.get("eligible_ids", ()))
            and all(
                saved.get("groups", {}).get(k) == v for k, v in current_groups.items()
            )
        )
        legacy_reuse = (
            not author_batch
            and plan["execution"] == "fixed"
            and saved.get("ready")
            and any(e.get("attempts") for e in state["entries"].values())
        )
        if reusable or legacy_reuse:
            calibration = saved
        else:
            calibration = calibrate(
                root, {**plan, "entries": eligible}, display=display
            )
            calibration["frozen_plan_sha256"] = digest(plan)
            calibration["eligible_ids"] = sorted(current_groups)
            write_json(saved_calibration, calibration)
        if recommend_only:
            state["status"] = "recommended"
            write_json(root / "batch.json", state)
            return {
                "directory": str(root),
                "status": "recommended",
                "recommendations": calibration["recommendations"],
                "experiments": [
                    {
                        "experiment_id": e["experiment_id"],
                        "profile": calibration["recommendations"][
                            calibration["groups"][e["experiment_id"]]
                        ],
                    }
                    for e in eligible
                ],
                "reason": calibration["reason"],
                "active_seconds": calibration["active_seconds"],
                "waiting_seconds": calibration["waiting_seconds"],
                "wall_seconds": calibration["wall_seconds"],
                "calibrated": calibration["calibrated"],
                "uncalibrated_groups": calibration["uncalibrated_groups"],
            }
        import ray

        from smartsom.experiments.composable import implementation_identity
        from smartsom.experiments.tuning_broker import AdaptiveBroker
        from smartsom.experiments.tuning_callbacks import EvidenceCallback
        from smartsom.experiments.tuning_ray import build_tuner

        if (
            not (
                runtime_source_matches(plan["source"], source_identity())
                if author_batch
                else source_identity() == plan["source"]
            )
            or implementation_identity() != plan["implementation_sha256"]
        ):
            raise ValueError("batch source/dependencies changed during calibration")
        if ray.is_initialized():
            raise RuntimeError(
                "batch runner requires ownership of a fresh local Ray runtime"
            )
        entries, profiles, selected = [], {}, {}
        for frozen in plan["entries"]:
            identity = frozen["experiment_id"]
            previous = state["entries"][identity]
            if previous["status"] == "completed" or (
                previous["status"] == "failed" and not retry_failed
            ):
                continue
            item = dict(frozen)
            prepared = PreparedComposition(
                **previous.get("selected_prepared", item["prepared"])
            )
            item["prepared"] = asdict(prepared)
            config = prepared.config
            attempt = len(previous.get("attempts", [])) + 1
            item["run_dir"] = str(
                root / "experiments" / identity / f"attempt-{attempt:04d}"
            )
            Path(item["run_dir"]).mkdir(parents=True)
            continuation = previous.get("checkpoint")
            if continuation:
                old_record = json.loads(
                    (Path(continuation) / "record.json").read_text()
                )
                verify_identity(prepared, old_record, continuation)
                item["record"] = old_record
            else:
                item["record"] = {
                    "schema": (
                        "smartsom.experiment/v4"
                        if prepared.config.schema_id == "smartsom.execution-config/v2"
                        else "smartsom.experiment/v3"
                    ),
                    "kind": "training",
                    "provider": "composable",
                    "id": identity,
                    "status": "running",
                    "source": plan["source"],
                    "implementation_sha256": plan["implementation_sha256"],
                    "scientific_sha256": prepared.scientific_sha256,
                }
            item["continuation"] = continuation
            overhead = config.runtime.sampling_processes
            item["execution_contract"] = {
                "schema": "smartsom.tune-execution/v1",
                "cpu_overhead": overhead,
                "sampling_child_threads": 1,
                "allocation_epoch": previous.get("allocation_epoch", 0),
            }
            item["remaining_updates"] = math_ceil_div(
                config.training.total_ticks, config.training.ticks_per_update
            )
            group = calibration["groups"][identity]
            item["calibration_group"] = group
            recommended = (
                previous.get("selected_profile")
                or calibration["recommendations"][group]
            )
            measured = [
                m
                for m in calibration["measurements"]
                if m["group"] == group and m["valid"] and m["throughput"] > 0
            ]
            options = []
            by_profile = {}
            for measurement in measured:
                p = measurement["profile"]
                if (p["device"] == "cpu") != (config.runtime.device == "cpu"):
                    continue
                if (p.get("num_envs", 1), p.get("sampling_processes", 0)) == (
                    recommended.get("num_envs", 1),
                    recommended.get("sampling_processes", 0),
                ):
                    by_profile.setdefault(
                        (p["threads"], p["concurrency"], p["device"]), []
                    ).append(measurement)
            for rows in by_profile.values():
                p = rows[0]["profile"]
                options.append(
                    {
                        "CPU": p["threads"] + p.get("sampling_processes", 0),
                        "GPU": int(config.runtime.device == "cuda"),
                        "memory": max(
                            1,
                            max(
                                int(
                                    m["stages"].get(
                                        "per_trial_peak_memory",
                                        math_ceil_div(
                                            m["peak_memory"], p["concurrency"]
                                        ),
                                    )
                                )
                                for m in rows
                            ),
                        ),
                        "gpu_memory": max(m["peak_gpu_memory"] for m in rows),
                        "concurrency": p["concurrency"],
                        "throughput": statistics.median(m["throughput"] for m in rows),
                        "update_seconds": max(
                            max(
                                0.001,
                                m["elapsed_seconds"] - m["stages"].get("cold_start", 0),
                            )
                            for m in rows
                        ),
                        "restart_seconds": max(
                            m["stages"].get("cold_start", m["elapsed_seconds"])
                            for m in rows
                        ),
                        "profile": p,
                    }
                )
            chosen = next(
                (p for p in options if p["profile"] == recommended),
                previous.get("selected_option"),
            )
            if chosen is None and group in calibration.get("uncalibrated_groups", []):
                chosen = {
                    "CPU": recommended["threads"]
                    + recommended.get("sampling_processes", 0),
                    "GPU": int(config.runtime.device == "cuda"),
                    "memory": 1,
                    "gpu_memory": 0,
                    "concurrency": recommended["concurrency"],
                    "throughput": 0.0,
                    "update_seconds": 1.0,
                    "restart_seconds": 1.0,
                    "profile": recommended,
                    "uncalibrated": True,
                }
            if chosen is None:
                raise ValueError(
                    "selected calibration profile lacks a valid measurement"
                )
            if not continuation:
                selected_prepared = _with_selected_layout(prepared, recommended)
                item["prepared"] = asdict(selected_prepared)
                item["record"]["scientific_sha256"] = (
                    selected_prepared.scientific_sha256
                )
                previous["selected_prepared"] = item["prepared"]
                previous["selected_profile"] = recommended
                previous["selected_option"] = chosen
            item["execution_contract"]["cpu_overhead"] = recommended.get(
                "sampling_processes", 0
            )
            item["record"]["calibration"] = {
                "profile": recommended,
                "calibrated": group not in calibration.get("uncalibrated_groups", []),
                "schedule_status": (calibration.get("schedule") or {}).get(
                    "status", "schedule_uncalibrated"
                ),
                "candidate_source": calibration.get("historical_candidates", {})
                .get(group, {})
                .get("report"),
                "converged": group in calibration.get("converged_groups", []),
                "wall_seconds": calibration.get("wall_seconds"),
            }
            options = [p for p in options if p is not chosen]
            profiles[identity] = [chosen, *options]
            selected[identity] = {key: chosen[key] for key in ("CPU", "GPU", "memory")}
            previous.setdefault("attempts", []).append(
                {"run_dir": item["run_dir"], "started_at": time.time()}
            )
            previous["status"] = "queued"
            entries.append(item)
            if display:
                from smartsom.telemetry.workflow import describe_prepared

                selected_prepared = PreparedComposition(**item["prepared"])
                display.update(
                    identity,
                    {
                        "status": "queued",
                        "stage": "queued",
                        "workflow": describe_prepared(
                            selected_prepared, "train-evaluate"
                        ),
                        "display_name": _algorithm_display_name(
                            selected_prepared, identity
                        ),
                    },
                    total=selected_prepared.config.training.total_ticks,
                    unit="physical ticks",
                )
        snapshot = ResourceMonitor().snapshot(exclude_pids=(os.getpid(),))
        try:
            # Ray 2.58's uv hook overrides even an explicit py_executable and
            # runs workers from a copied working directory with a new empty
            # environment. The local batch owns Ray and uses this installed venv.
            os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
            ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
            _framework_call(
                ray.init,
                address="local",
                num_cpus=int(snapshot.cpus),
                num_gpus=len(snapshot.gpus),
                include_dashboard=False,
                log_to_driver=False,
                runtime_env={"py_executable": sys.executable},
            )
            state["status"] = "running"
            write_json(root / "batch.json", state)
            if display:
                display.configure_tuning(
                    {
                        "stage": "running",
                        "calibration": {
                            "level": calibration.get("calibration_level", "quick"),
                            "calibrated": calibration["calibrated"],
                            "schedule_status": (calibration.get("schedule") or {}).get(
                                "status", "schedule_uncalibrated"
                            ),
                            "historical_candidates": calibration.get(
                                "historical_candidates", {}
                            ),
                        },
                    }
                )
            cohorts = []
            for device in ("cpu", "cuda"):
                cohort = [
                    e
                    for e in entries
                    if PreparedComposition(**e["prepared"]).config.runtime.device
                    == device
                ]
                cohorts.append((device, cohort))
            for device, cohort in cohorts:
                if not cohort:
                    continue
                broker = AdaptiveBroker(
                    cohort,
                    {e["experiment_id"]: profiles[e["experiment_id"]] for e in cohort},
                    mode=plan["mode"],
                    execution=plan["execution"],
                    global_limit=max(
                        (e.get("baseline_concurrency", 1) for e in cohort), default=1
                    )
                    if author_batch
                    else None,
                    file_limits={
                        e.get("file_id", e["experiment_id"].split("__", 1)[0]): e.get(
                            "baseline_concurrency", 1
                        )
                        for e in cohort
                    }
                    if author_batch
                    else None,
                    incompatible_group_pairs=(calibration.get("schedule") or {}).get(
                        "incompatible_pairs", []
                    ),
                    uncalibrated_groups=calibration.get("uncalibrated_groups", []),
                )
                segment = f"segment-{len(state['segments']) + 1:04d}-{device}"
                state["segments"].append(
                    {
                        "id": segment,
                        "entries": [e["experiment_id"] for e in cohort],
                        "status": "running",
                    }
                )
                write_json(root / "batch.json", state)
                grid = _framework_call(
                    build_tuner(
                        cohort,
                        selected={
                            e["experiment_id"]: selected[e["experiment_id"]]
                            for e in cohort
                        },
                        broker=broker,
                        storage_path=root / "ray",
                        name=segment,
                        device=device,
                        allocation_function=broker.allocation,
                        callbacks=(EvidenceCallback(root, cohort, broker),),
                    ).fit
                )
                from smartsom.experiments.control import boundary

                boundary(root)
                state = json.loads((root / "batch.json").read_text())
                state["segments"][-1]["status"] = (
                    "failed" if grid.errors else "completed"
                )
                write_json(root / "batch.json", state)
            state["status"] = (
                "completed"
                if all(e["status"] == "completed" for e in state["entries"].values())
                else "failed"
            )
        finally:
            ray.shutdown()
    except BaseException as exc:
        state = json.loads((root / "batch.json").read_text())
        recover_committed(root, plan, state)
        if isinstance(exc, KeyboardInterrupt):
            for entry in state["entries"].values():
                if entry.get("status") == "running":
                    entry["status"] = "interrupted"
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure={"exception": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        write_json(root / "batch.json", state)
    return {
        "directory": str(root),
        "status": state["status"],
        "completed": sum(e["status"] == "completed" for e in state["entries"].values()),
        "failed": sum(e["status"] == "failed" for e in state["entries"].values()),
    }


def math_ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def run_batch(
    *,
    batch=None,
    study=None,
    action="run",
    mode=None,
    execution=None,
    calibration_timeout=None,
    preflight=None,
    preflight_coverage=None,
    display=None,
):
    from dataclasses import replace

    if action not in {"check", "recommend", "run"}:
        raise ValueError("Tune action must be check, recommend or run")
    inputs = load_batch(batch, study)
    inputs = replace(
        inputs,
        **{
            key: value
            for key, value in (
                ("mode", mode),
                ("execution", execution),
                ("active_limit", calibration_timeout),
                ("preflight", preflight),
                ("preflight_coverage", preflight_coverage),
            )
            if value is not None
        },
    )
    checked = check_batch_inputs(inputs)
    if action == "check":
        return checked
    root, plan, state = allocate_batch(inputs)
    return execute_batch(
        root, plan, state, recommend_only=action == "recommend", display=display
    )


def resume_batch(directory, *, retry_failed=False, display=None):
    from smartsom.experiments.batch import exclusive_lock

    root = Path(directory).expanduser().resolve()
    from smartsom.telemetry.runtime import bind

    bind(root)
    with exclusive_lock(root / "driver.lock"):
        root, plan, state = load_run(root)
        if (plan.get("provenance") or {}).get("kind") != "author-plan":
            from smartsom.experiments.preflight import run as run_preflight

            run_preflight(root, plan)
        return _execute_batch(
            root, plan, state, retry_failed=retry_failed, display=display
        )


def execute_batch(root, plan, state, **kwargs):
    from smartsom.experiments.batch import exclusive_lock
    from smartsom.telemetry.runtime import bind

    bind(root)
    with exclusive_lock(Path(root) / "driver.lock"):
        if (plan.get("provenance") or {}).get("kind") != "author-plan":
            from smartsom.experiments.preflight import run as run_preflight

            run_preflight(root, plan)
        return _execute_batch(root, plan, state, **kwargs)


def recover_committed(root, plan, state):
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.experiments.tuning_session import verify_identity

    for entry in plan["entries"]:
        row = state["entries"][entry["experiment_id"]]
        for attempt in row.get("attempts", ()):
            directory = Path(attempt["run_dir"]).resolve()
            if not directory.is_relative_to(Path(root).resolve()):
                raise ValueError("attempt path escapes its batch")
            pointer = directory / "checkpoints/adaptive-recovery.json"
            if not pointer.exists():
                continue
            checkpoint = _contained(
                directory / "checkpoints", json.loads(pointer.read_text())["checkpoint"]
            )
            record = json.loads((checkpoint / "record.json").read_text())
            marker = verify_identity(
                PreparedComposition(**row.get("selected_prepared", entry["prepared"])),
                record,
                checkpoint,
            )
            if marker["physical_ticks"] >= row.get("physical_ticks", 0):
                row.update(
                    checkpoint=str(checkpoint),
                    commit_id=marker["commit_id"],
                    physical_ticks=marker["physical_ticks"],
                    updates=marker["updates"],
                    allocation_epoch=marker["allocation_epoch"],
                )
                if marker["phase"] == "experiment_complete":
                    row["status"] = "completed"
