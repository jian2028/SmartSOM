"""Local advisory profiles, with separate probe and online-training evidence."""

import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments.batch import exclusive_lock
from smartsom.experiments.evidence import write_json
from smartsom.experiments.tuning_resources import ExecutionProfile

SCHEMA = "smartsom.performance-profiles/v1"
PROBE_VERSION = "v4-disposable-update-validation/2"
ONLINE_SCHEMA = "smartsom.online-performance/v1"


def cache_root(output_root):
    path = Path(output_root).resolve()
    runs = next((p for p in (path, *path.parents) if p.name == "runs"), None)
    if runs is None:
        runs = path.parent
    return runs / ".performance-profiles"


def hardware_shape(snapshot, *, mode):
    return digest(
        {
            "host": platform.node(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "cpus": snapshot.cpus,
            "memory_total": snapshot.memory_total,
            "limits": snapshot.limits,
            "mode": mode,
            "gpus": [(g.index, g.memory_total) for g in snapshot.gpus],
        }
    )


def group_shape(group, *, level):
    source = group.get("source_prepared", group["prepared"])
    config = json.loads(source["config_json"])
    training = config["training"]
    validation = json.loads(source["validation_json"])
    scenario = json.loads(source["scenario_json"])
    factory = scenario.get("factory", {})
    cases = [row.get("scenario", {}) for row in validation]
    demands = scenario.get("demands", [])
    steps = [step for demand in demands for step in demand.get("steps", [])]
    return digest(
        {
            "probe": PROBE_VERSION,
            "level": level,
            "backend": training.get("backend"),
            "algorithm": training.get("algorithm"),
            "groups": training.get("groups"),
            "mode": training.get("mode"),
            "gamma": training.get("gamma"),
            "reward": training.get("reward"),
            "parameters": source.get("parameters_json"),
            "runtime_layout": config["runtime"] if level == "online" else None,
            "policies": json.loads(source["policies_json"]),
            "device": config["runtime"]["device"],
            "factory": {
                "machines": len(factory.get("machines", [])),
                "buffers": len(factory.get("buffers", [])),
                "agvs": len(factory.get("agvs", [])),
                "topology": digest(factory),
            },
            "workload": {
                "demands": len(demands),
                "steps": len(steps),
                "nominal_work": sum(
                    float(step.get("nominal_ticks", 0)) for step in steps
                ),
                "operation_types": sorted(
                    {step.get("operation_type") for step in steps}
                ),
                "case_demands": [len(case.get("demands", [])) for case in cases],
                "case_ticks": [case.get("tick_limit") for case in cases],
            },
            "update": training.get("ticks_per_update"),
            "episode": training.get("max_ticks"),
            "validation": config.get("validation"),
            "checkpointing": config.get("checkpointing"),
            "matching": json.loads(source["composition_json"]).get("matching"),
        }
    )


def _valid_row(row, *, hardware, shape):
    try:
        if row["hardware"] != hardware or row["shape"] != shape:
            return False
        profile = ExecutionProfile(**row["profile"])
        valid = (
            asdict(profile) == row["profile"]
            and math.isfinite(row["throughput"])
            and row["throughput"] > 0
        )
        if not valid:
            return False
        report = json.loads(Path(row["report"]).read_text())
        return report.get("schema") == "smartsom.tune-calibration/v1" and any(
            item.get("profile") == row["profile"]
            and item.get("throughput") == row["throughput"]
            and item.get("valid") is True
            and item.get("termination") is None
            for item in report.get("measurements", ())
        )
    except (KeyError, ValueError, TypeError, OverflowError, OSError):
        return False


def select(output_root, *, hardware, shape, candidate):
    root = cache_root(output_root)
    if candidate not in {"latest", "best"}:
        report = Path(candidate).expanduser().resolve()
        try:
            payload = json.loads(report.read_text())
            rows = (
                payload["profiles"]
                if payload.get("schema") == SCHEMA
                else payload["profile_records"]
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid calibration candidate report: {report}") from exc
    else:
        try:
            payload = json.loads((root / "index.json").read_text())
            rows = payload["profiles"] if payload["schema"] == SCHEMA else []
        except (OSError, ValueError, KeyError, TypeError):
            rows = []
    compatible = [
        row for row in rows if _valid_row(row, hardware=hardware, shape=shape)
    ]
    if candidate not in {"latest", "best"} and not compatible:
        raise ValueError(
            "specified calibration report has no shape-compatible valid candidate"
        )
    if not compatible:
        return None
    key = (
        (lambda row: (row["throughput"], row["at"]))
        if candidate == "best"
        else (lambda row: row["at"])
    )
    return max(compatible, key=key)


def store(output_root, records):
    if not records:
        return
    root = cache_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "index.lock"):
        path = root / "index.json"
        try:
            old = json.loads(path.read_text())
            rows = old["profiles"] if old["schema"] == SCHEMA else []
        except (OSError, ValueError, KeyError, TypeError):
            rows = []
        write_json(
            path,
            {
                "schema": SCHEMA,
                "updated_at": time.time(),
                "profiles": [*rows, *records],
            },
        )


def online_shape(group, source, implementation):
    return digest(
        {
            "task": group_shape(group, level="online"),
            "implementation": implementation,
            "python": source.get("python"),
            "packages": source.get("packages"),
        }
    )


def select_online(output_root, *, hardware, shape):
    """Only a row backed by a retained online report may supply a warm hint."""
    try:
        index = json.loads((cache_root(output_root) / "online.json").read_text())
        rows = index["profiles"] if index["schema"] == ONLINE_SCHEMA else []
        compatible = []
        for row in rows:
            if row["hardware"] != hardware or row["shape"] != shape:
                continue
            report = json.loads(Path(row["report"]).read_text())
            if (
                report.get("schema") == ONLINE_SCHEMA
                and row in report.get("profiles", [])
                and type(row["concurrency"]) is int
                and row["concurrency"] > 0
                and math.isfinite(row["throughput"])
                and row["throughput"] > 0
            ):
                compatible.append(row)
        return max(compatible, key=lambda row: row["at"], default=None)
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return None


def store_online(output_root, records):
    """Keep the latest measured operating point per allocated machine/task."""
    if not records:
        return
    root = cache_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "index.lock"):
        path = root / "online.json"
        try:
            payload = json.loads(path.read_text())
            old = payload["profiles"] if payload["schema"] == ONLINE_SCHEMA else []
        except (OSError, ValueError, KeyError, TypeError):
            old = []
        merged = {(row["hardware"], row["shape"]): row for row in old}
        merged.update({(row["hardware"], row["shape"]): row for row in records})
        write_json(path, {"schema": ONLINE_SCHEMA, "profiles": list(merged.values())})
