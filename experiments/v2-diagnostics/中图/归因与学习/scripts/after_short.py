"""Wait for reviewed short checks, then invoke the official long-run command.

This does not execute cases, manage workers, or change a running experiment.
"""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PACKAGE = Path(__file__).resolve().parents[1]
ARMS = ("base-original", "base-window", "sl-original", "sl-window")


def verify_short(root):
    panels = {}
    reference_worlds = None
    for arm in ARMS:
        trial = root / "trials" / ("check-" + arm) / "seed-101"
        episodes = [
            json.loads(s) for s in (trial / "episodes.jsonl").read_text().splitlines()
        ]
        if len(episodes) != 2:
            raise ValueError(f"{arm}: expected two committed episodes")
        for row in episodes:
            audit = row["losses"]["v2_tail_audit"]
            reason = row["termination_reason"]
            expected = reason == "all_delivered" or (
                arm.endswith("window") and reason == "drain_deadline"
            )
            if (
                audit["end_reason"] != reason
                or audit["terminal_used_by_learner"] != expected
            ):
                raise ValueError(f"{arm}: boundary provenance mismatch")
            if expected and audit["actual_omitted_transitions"] != 0:
                raise ValueError(f"{arm}: unexpected terminal omissions")
        for node in (0, 1, 2):
            path = trial / "periodic_evaluation" / f"episode-{node:08d}" / "argmax"
            rows = [
                json.loads(s) for s in (path / "cases.jsonl").read_text().splitlines()
            ]
            if len(rows) != 12 or not (path / "summary.json").is_file():
                raise ValueError(f"{arm}: incomplete panel at {node}")
            worlds = {
                (r["scenario_seed"], r["action_seed"], r["condition"]): r[
                    "evaluation_identity"
                ]["world_sha256"]
                for r in rows
            }
            if len(worlds) != 12:
                raise ValueError(f"{arm}: duplicate cases at {node}")
            if reference_worlds is None:
                reference_worlds = worlds
            elif worlds != reference_worlds:
                raise ValueError(f"{arm}: changed fixed panel at {node}")
            fields = (
                "scenario_seed",
                "action_seed",
                "condition",
                "completed",
                "failed",
                "tick",
                "joint_cost",
                "observed_return",
            )
            if node == 0:
                panels[arm] = sorted(
                    (tuple(r[k] for k in fields) for r in rows), key=lambda r: r[:3]
                )
    for model in ("base", "sl"):
        if panels[model + "-original"] != panels[model + "-window"]:
            raise ValueError(f"{model}: initial paired results differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument(
        "--once", action="store_true", help="inspect once; never launch"
    )
    args = parser.parse_args()
    short, target = args.short.resolve(), args.long.resolve()
    cli = [sys.executable, "-m", "smartsom.experiments.cli"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = subprocess.run(
                cli + ["status", str(short), "--json"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
            status = json.loads(result.stdout)["status"]
            print(json.dumps({"short_status": status, "long": str(target)}), flush=True)
            if status == "completed":
                verify_short(short)
                if args.once:
                    return
                if target.exists():
                    raise FileExistsError(
                        f"refusing to relaunch existing output {target}"
                    )
                command = cli + ["run", str(PACKAGE)]
                for arm in ARMS:
                    command += ["--id", "long-" + arm]
                command += [
                    "--max-concurrent",
                    "4",
                    "--evaluation-processes",
                    "1",
                    "--evaluation-cpus",
                    "1",
                    "--cpus",
                    "8",
                    "--memory-gib",
                    "64",
                    "--wandb",
                    "offline",
                    "--output-root",
                    str(target),
                    "--background",
                ]
                print(json.dumps({"launch_command": command}), flush=True)
                subprocess.run(command, cwd=REPO, check=True)
                return
            if status not in {"running", "queued", "prepared"}:
                raise RuntimeError(f"short checks are {status}; long runs not launched")
            if args.once:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
