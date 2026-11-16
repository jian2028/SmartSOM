"""Qualify an idle native acceptance host without changing application policy."""

import argparse
import json
import os
import time
from pathlib import Path

from smartsom.telemetry.system_resources import GIB, ResourceMonitor


def qualify(
    monitor,
    *,
    timeout=600,
    interval=5,
    consecutive=3,
    mode="balanced",
    required_cpus=1.25,
    native_memory=0,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Fail rather than run acceptance on an unavailable host or retry a test."""
    if mode not in {"balanced", "performance"}:
        raise ValueError("unknown CI host qualification mode")
    started, stable = clock(), 0
    history = []
    while clock() - started <= timeout:
        snapshot = monitor.snapshot(exclude_pids=(os.getpid(),))
        # Test-host qualification only: retain the pre-migration thresholds.
        # No application admission, leases or performance calibration use this.
        cpu_reserve = max(1.0, snapshot.cpus * (0.2 if mode == "balanced" else 0.05))
        memory_reserve = (
            max(2 * GIB, int(snapshot.memory_total * 0.2))
            if mode == "balanced"
            else max(GIB, int(snapshot.memory_total * 0.1))
        )
        external = (
            snapshot.external_cpu_load
            if snapshot.external_cpu_load is not None
            else snapshot.cpu_load
        )
        capacity = {
            "cpus": max(0.0, snapshot.cpus * (1 - external / 100) - cpu_reserve),
            "memory": max(
                0,
                min(
                    snapshot.memory_total,
                    snapshot.memory_available
                    + sum(p.rss for p in snapshot.processes if p.excluded),
                )
                - memory_reserve,
            ),
        }
        native_capacity = None
        if native_memory:
            native_capacity = {
                "cpus": max(0, snapshot.cpus * (1 - snapshot.cpu_load / 100) - 0.25),
                "memory": max(
                    0,
                    snapshot.memory_available
                    - max(2 * GIB, int(snapshot.memory_total * 0.2)),
                ),
            }
        eligible = (
            snapshot.external_cpu_load is not None
            and capacity["cpus"] >= required_cpus
            and capacity["memory"] >= GIB // 4
            and (native_capacity is None or native_capacity["memory"] >= native_memory)
        )
        stable = stable + 1 if eligible else 0
        history.append(
            {
                "at": clock() - started,
                "cpus": snapshot.cpus,
                "external_percent": snapshot.external_cpu_load,
                "memory_total": snapshot.memory_total,
                "memory_available": snapshot.memory_available,
                "capacity": capacity,
                "native_capacity": native_capacity,
                "stable_samples": stable,
                "raw_cpu": snapshot.observations,
            }
        )
        history = history[-120:]
        if stable >= consecutive:
            return {"qualified": True, "samples": history}
        sleep(interval)
    return {"qualified": False, "samples": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("balanced", "performance"), default="balanced"
    )
    parser.add_argument("--required-cpus", type=float, default=1.25)
    args = parser.parse_args()
    if args.required_cpus <= 0:
        parser.error("--required-cpus must be positive")
    result = qualify(
        ResourceMonitor(), mode=args.mode, required_cpus=args.required_cpus
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"qualified": result["qualified"], "samples": len(result["samples"])}
        )
    )
    if not result["qualified"]:
        raise SystemExit("host never supplied the required stable acceptance capacity")


if __name__ == "__main__":
    main()
