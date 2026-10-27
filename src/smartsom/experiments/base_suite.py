"""One bounded, clean-source overnight diagnostic; no external scheduler needed."""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from smartsom.config.base_diagnostic import load_inputs
from smartsom.experiments.base_diagnostic import source_identity, write_json


class Suite:
    def __init__(self, root, hours=24, workers=3):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600
        self.workers = workers
        self.processes = {}
        self.jobs = {}
        self.lock = threading.Lock()
        self.phase = "initializing"
        self.source = source_identity()
        if self.source["dirty"]:
            raise ValueError("suite requires a clean committed checkout")
        write_json(
            self.root / "manifest.json",
            {
                "source": self.source,
                "pid": os.getpid(),
                "start_unix": time.time(),
                "hours": hours,
                "training_workers": workers,
                "training_seeds": [101, 102, 103],
                "evidence": "development-diagnostic",
                "checkpoint_rule": "fixed final checkpoint; no selection on test cases",
                "algorithm_scope": "initial PPO route; value-learning comparison not launched",
            },
        )

    def publish(self):
        write_json(
            self.root / "progress.json",
            {
                "phase": self.phase,
                "pid": os.getpid(),
                "elapsed_seconds": time.monotonic() - self.started,
                "remaining_seconds": max(0, self.deadline - time.monotonic()),
                "jobs": self.jobs,
            },
        )

    def run_job(self, name, arguments, seconds):
        output = self.root / name
        command = [
            sys.executable,
            "-m",
            "smartsom.experiments.base_diagnostic",
            *map(str, arguments),
            "--output",
            str(output),
        ]
        limit = min(self.deadline, time.monotonic() + seconds)
        if limit <= time.monotonic() or (self.root / "STOP").exists():
            with self.lock:
                self.jobs[name] = {
                    "status": "not_started_budget_or_stop",
                    "output": str(output),
                }
                self.publish()
            return self.jobs[name]
        with (self.root / (name + ".log")).open("w") as stream:
            child = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            with self.lock:
                self.processes[name] = child
                self.jobs[name] = {
                    "status": "running",
                    "pid": child.pid,
                    "output": str(output),
                    "command": command,
                }
                self.publish()
            status = "running"
            while child.poll() is None:
                if time.monotonic() >= limit or (self.root / "STOP").exists():
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    status = (
                        "stopped"
                        if (self.root / "STOP").exists()
                        else "wall_budget_reached"
                    )
                    break
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if status == "running":
                status = "completed" if child.returncode == 0 else "failed"
                progress_path = output / "progress.json"
                if status == "completed" and progress_path.is_file():
                    progress = json.loads(progress_path.read_text())
                    if progress.get("status") == "wall_budget_reached":
                        status = "wall_budget_reached"
            with self.lock:
                self.jobs[name].update(status=status, exit_code=child.returncode)
                self.processes.pop(name, None)
                self.publish()
        return self.jobs[name]

    def wave(self, phase, jobs, workers):
        self.phase = phase
        self.publish()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.run_job, *job): job[0] for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def stop(self, *_):
        (self.root / "STOP").touch()

    def execute(self, config_root):
        config_root = Path(config_root).resolve()
        experiments = {
            scale: config_root / f"{scale}.yaml" for scale in ("small", "target")
        }
        train_seeds = {
            scale: load_inputs(path)[0].train_seeds
            for scale, path in experiments.items()
        }
        calibration = {
            scale: self.root / f"calibrate-{scale}" / "calibration.json"
            for scale in experiments
        }
        jobs = [
            (f"calibrate-{s}", ["calibrate", ex, "--seconds", 3500, "--cases", 2], 3600)
            for s, ex in experiments.items()
        ]
        self.wave("calibration", jobs, 2)
        if any(
            not p.exists() or json.loads(p.read_text())["status"] != "frozen"
            for p in calibration.values()
        ):
            raise RuntimeError(
                "calibration incomplete: learning comparisons not started"
            )
        jobs = [
            (
                f"pilot-{s}",
                [
                    "train",
                    ex,
                    "--calibration",
                    calibration[s],
                    "--seed",
                    991,
                    "--ticks",
                    4000,
                    "--seconds",
                    900,
                ],
                930,
            )
            for s, ex in experiments.items()
        ]
        self.wave("throughput_pilots", jobs, 2)
        budgets = {}
        for scale, hours in (("small", 12), ("target", 6)):
            progress = json.loads(
                (self.root / f"pilot-{scale}" / "progress.json").read_text()
            )
            if (
                progress.get("physical_ticks") != 4000
                or progress.get("updates", 0) == 0
            ):
                raise RuntimeError("pilot did not perform its complete learning budget")
            rate = 4000 / progress["wall_seconds"]
            # A full rollout grows WIP/candidate sets beyond the pilot. Keep a
            # conservative allowance for that change and 3-way CPU contention.
            budgets[scale] = max(4000, int(rate * hours * 3600 * 0.45 / 1000) * 1000)
        write_json(
            self.root / "budgets.json",
            {
                "physical_ticks_per_seed": budgets,
                "pilot_ticks": 4000,
                "safety_fraction": 0.45,
                "concurrency": self.workers,
                "note": "identical per-scale seed budgets; wall-limited incomplete jobs are explicit",
            },
        )
        for scale, hours in (("small", 12), ("target", 6)):
            jobs = [
                (
                    f"train-{scale}-{seed}",
                    [
                        "train",
                        experiments[scale],
                        "--calibration",
                        calibration[scale],
                        "--seed",
                        seed,
                        "--ticks",
                        budgets[scale],
                        "--seconds",
                        hours * 3600 - 60,
                    ],
                    hours * 3600,
                )
                for seed in train_seeds[scale]
            ]
            self.wave(f"train_{scale}", jobs, self.workers)
        jobs = []
        for scale, ex in experiments.items():
            for split in ("validation", "test"):
                jobs.append(
                    (
                        f"{split}-{scale}-rules",
                        [
                            "evaluate",
                            ex,
                            "--calibration",
                            calibration[scale],
                            "--split",
                            split,
                            "--method",
                            "rule",
                        ],
                        4 * 3600,
                    )
                )
                jobs.append(
                    (
                        f"{split}-{scale}-random",
                        [
                            "evaluate",
                            ex,
                            "--calibration",
                            calibration[scale],
                            "--split",
                            split,
                            "--method",
                            "random",
                            "--action-seeds",
                            3 if split == "test" else 1,
                        ],
                        4 * 3600,
                    )
                )
                for seed in train_seeds[scale]:
                    checkpoint = self.root / f"train-{scale}-{seed}" / "last.pt"
                    if not checkpoint.exists():
                        continue
                    for selection in ("argmax", "sample"):
                        jobs.append(
                            (
                                f"{split}-{scale}-ppo-{seed}-{selection}",
                                [
                                    "evaluate",
                                    ex,
                                    "--calibration",
                                    calibration[scale],
                                    "--split",
                                    split,
                                    "--method",
                                    "ppo",
                                    "--checkpoint",
                                    checkpoint,
                                    "--selection",
                                    selection,
                                    "--action-seeds",
                                    3
                                    if split == "test" and selection == "sample"
                                    else 1,
                                ],
                                4 * 3600,
                            )
                        )
        self.wave("paired_evaluation", jobs, min(8, os.cpu_count() or 1))
        summaries = {
            name: json.loads((self.root / name / "summary.json").read_text())
            for name in self.jobs
            if (self.root / name / "summary.json").is_file()
        }
        write_json(self.root / "comparison.json", summaries)
        self.phase = (
            "completed"
            if all(j["status"] == "completed" for j in self.jobs.values())
            else "completed_with_incomplete_jobs"
        )
        self.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.hours <= 0 or not 1 <= args.workers <= 4:
        raise ValueError("positive wall budget and 1..4 training workers required")
    suite = Suite(args.output, args.hours, args.workers)
    signal.signal(signal.SIGTERM, suite.stop)
    signal.signal(signal.SIGINT, suite.stop)
    try:
        suite.execute(args.config_root)
    except BaseException as exc:
        suite.phase = "failed"
        suite.publish()
        write_json(
            suite.root / "failure.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
