"""Qualify an idle native acceptance host without changing application policy."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from smartsom.experiments.tuning_resources import GIB, ResourceBroker, ResourceMonitor


def qualify(
    monitor,
    *,
    timeout=600,
    interval=5,
    consecutive=3,
    mode="balanced",
    required_cpus=1.25,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Fail rather than run acceptance on an unavailable host or retry a test."""
    manager = ResourceBroker(monitor, mode=mode)
    started, stable = clock(), 0
    history = []
    while clock() - started <= timeout:
        snapshot = monitor.snapshot(exclude_pids=(os.getpid(),))
        capacity = manager.capacity(snapshot)
        eligible = (
            snapshot.external_cpu_load is not None
            and capacity.cpus >= required_cpus
            and capacity.memory >= GIB // 4
        )
        stable = stable + 1 if eligible else 0
        history.append(
            {
                "at": clock() - started,
                "cpus": snapshot.cpus,
                "external_percent": snapshot.external_cpu_load,
                "memory_total": snapshot.memory_total,
                "memory_available": snapshot.memory_available,
                "capacity": asdict(capacity),
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
