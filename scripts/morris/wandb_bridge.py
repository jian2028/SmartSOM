"""Mirror a SmartSOM v4 batch run's on-disk progress into Weights & Biases.

The composable/v4 training path never initialises W&B itself (only
production_training.TrainingDisplay does, and that path is not used), so this
reads the run's own progress.json and training reports and logs them instead.
It changes nothing in the repository and never touches the running experiment.

Usage: python wandb_bridge.py RUN_DIRECTORY [--entity E] [--project P]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import wandb

POLL_SECONDS = 60


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def task_seed(task: dict) -> int | None:
    workflow = (task.get("values") or {}).get("workflow") or {}
    return workflow.get("training_seed")


def scalars(values: dict) -> dict:
    """Numeric fields worth a chart, flattened and prefixed by stage."""
    out = {}
    for key in (
        "physical_ticks",
        "updates",
        "optimization_steps",
        "training_deliveries",
        "validation_round",
        "validation_finished",
        "validation_tick",
    ):
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"train/{key}"] = value
    return out


def best_checkpoint(directory: Path) -> dict:
    best = load(directory / "checkpoints" / "best.json") or {}
    update = best.get("checkpoint", "")
    digits = "".join(c for c in str(update) if c.isdigit())
    return (
        {"train/best_update": int(digits), "train/best_ticks": best.get("physical_ticks", 0)}
        if digits
        else {}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--entity", default="creeps-usc")
    parser.add_argument("--project", default="SmartSOM2026")
    parser.add_argument("--label", default=None)
    # Reattach to existing W&B runs after a restart: "101=abc123,102=def456".
    parser.add_argument("--resume-ids", default="")
    args = parser.parse_args()

    root = args.run_directory.resolve()
    group = root.name
    runs: dict[int, wandb.sdk.wandb_run.Run] = {}
    sent: dict[int, int] = {}
    finished = set()
    resume_ids = {
        int(k): v
        for k, v in (pair.split("=") for pair in args.resume_ids.split(",") if pair)
    }

    print(f"bridging {root} -> {args.entity}/{args.project} (group {group})", flush=True)
    while True:
        progress = load(root / "logs" / "progress.json")
        if progress is None:
            time.sleep(POLL_SECONDS)
            continue

        for task in progress.get("tasks", []):
            values = task.get("values") or {}
            seed = task_seed(task)
            if seed is None:
                continue
            workflow = values.get("workflow") or {}
            if seed not in runs:
                runs[seed] = wandb.init(
                    entity=args.entity,
                    project=args.project,
                    group=group,
                    name=f"{args.label or workflow.get('name', 'run')}-seed{seed}",
                    # One process holds a run per seed; "create_new" keeps the
                    # earlier seeds' runs open instead of finishing them.
                    reinit="create_new",
                    id=resume_ids.get(seed),
                    resume="allow" if seed in resume_ids else None,
                    config={
                        "seed": seed,
                        "algorithm": values.get("algorithm"),
                        "factory": workflow.get("factory"),
                        "map": workflow.get("map"),
                        "total_ticks": workflow.get("training_total"),
                        "ticks_per_update": workflow.get("round_size"),
                        "validation_every": workflow.get("validation_every"),
                        "validation_cases": workflow.get("validation_cases"),
                        "mode": "central",
                        "source": str(root),
                    },
                )
                print(f"  seed {seed}: {runs[seed].url}", flush=True)

            step = int(values.get("physical_ticks") or 0)
            if step > sent.get(seed, -1):
                metrics = scalars(values)
                directory = values.get("training_run_directory")
                if directory:
                    metrics.update(best_checkpoint(Path(directory)))
                runs[seed].log(metrics, step=step)
                sent[seed] = step

            if task.get("status") in ("completed", "failed", "interrupted", "stopped", "force_stopped"):
                if seed not in finished:
                    runs[seed].summary["final_status"] = task.get("status")
                    runs[seed].finish()
                    finished.add(seed)
                    print(f"  seed {seed}: {task.get('status')}", flush=True)

        tasks = progress.get("tasks", [])
        if tasks and len(finished) == len(tasks):
            print("all tasks finished; bridge exiting", flush=True)
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
