"""Frozen nine-arm development comparison; separate from formal main/tag runs."""

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from smartsom.config.base_diagnostic import load_inputs, seed_for
from smartsom.experiments.social_checkpoint import (
    digest,
    read,
    reconcile,
    restore,
    save,
    write,
)

ROOT = Path(__file__).resolve().parents[3]
ARMS = (
    "base",
    "machine",
    "dispatcher-pickup",
    "dispatcher-delivery",
    "dispatcher",
    "pre",
    "post",
    "input",
    "all",
)
CONDITIONS = ("normal", "low", "medium", "high")
MILESTONES = (0, 100, 200, 300, 500, 750, 1000)


def inputs(manifest, arm):
    _, factory, workload, options, _ = load_inputs(ROOT / manifest["experiment"])
    return (
        factory,
        workload,
        options.model_copy(
            update={
                "social_arm": arm,
                "social_pool": "attention",
                "reconstructed_inputs": True,
                "reservation_stop": True,
                "empty_wait": True,
            }
        ),
    )


def verify(root):
    import smartsom

    manifest = read(root / "manifest.json")
    if not Path(smartsom.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("wrong imported source")
    for name, expected in manifest["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return manifest


def initialize(root, *, engineering=False):
    import torch

    if root.exists():
        raise ValueError("runroot already exists; choose a fresh directory")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty and not engineering:
        raise ValueError("scientific run requires final clean commit")
    experiment = (
        "configs/test/social/experiment.yaml"
        if engineering
        else "configs/base-diagnostic/target.yaml"
    )
    _, _, _, _, paths = load_inputs(ROOT / experiment)
    files = set((ROOT / "src/smartsom").rglob("*.py")) | set(paths.values())
    files |= {ROOT / experiment, ROOT / "uv.lock", ROOT / "pyproject.toml"}
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    manifest = {
        "schema": "smartsom.social-long/v1",
        "identity": root.name,
        "evidence": "engineering-probe"
        if engineering
        else "frozen-feature-development-comparison",
        "source": str(ROOT),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(dirty),
        "source_hashes": hashes,
        "experiment": experiment,
        "arms": list(ARMS),
        "seeds": [4101, 4102, 4103],
        "episodes": 1000,
        "block": 10,
        "early_ticks": 160000,
        "drain": 2000,
        "milestones": list(MILESTONES),
        "validation_worlds": 8,
        "test_worlds": 32,
        "reward_divisor": 1000,
        "created_unix": time.time(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "selection": "predeclared early-160k and ep1000; no automatic best test",
        "world_domain": "social-long-v1/world",
        "action_domain": "social-long-v1/action",
    }
    # The small recipe is an explicit separate output/manifest, never a budget override.
    if engineering:
        manifest.update(
            episodes=20,
            early_ticks=1,
            drain=50,
            milestones=[0, 20],
            validation_worlds=4,
            test_worlds=4,
        )
    root.mkdir(parents=True)
    for name in hashes:
        destination = root / "frozen-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    for arm in ARMS:
        _, _, options = inputs(manifest, arm)
        write(
            root / "configuration" / f"{arm}.json",
            options.model_dump(mode="json", by_alias=True),
        )
    write(root / "manifest.json", manifest)
    write(
        root / "control.json",
        {"workers": 4, "total_limit": 28, "pause_admission": False},
    )
    print(
        json.dumps(
            {
                "runroot": str(root),
                "commit": manifest["commit"],
                "evidence": manifest["evidence"],
            }
        )
    )


def compact(report):
    return {
        k: v
        for k, v in report.items()
        if k not in ("final_traffic_state", "final_station_state", "horizon_checks")
    }


def lock_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


@contextmanager
def worker_guard(root, directory):
    """Persist failures even if the parent controller has already disappeared."""
    with lock_file(directory / "worker.lock"):
        failure = root / "failures" / (directory.name + ".json")
        if failure.exists():
            raise RuntimeError(f"previous worker failure requires diagnosis: {failure}")
        try:
            yield
        except BaseException as error:
            write(
                failure,
                {
                    "pid": os.getpid(),
                    "directory": str(directory),
                    "unix": time.time(),
                    "error": repr(error),
                },
            )
            raise


def train(root, arm, seed, target):
    from smartsom.experiments.base_diagnostic import episode
    from smartsom.learning.base_ppo import BasePPO

    directory = root / "train" / f"{arm}-{seed}"
    with worker_guard(root, directory):
        manifest = verify(root)
        if arm not in manifest["arms"] or seed not in manifest["seeds"]:
            raise ValueError("run outside frozen panel")
        factory, workload, options = inputs(manifest, arm)
        policy = BasePPO(options, seed)
        path = directory / "last.pt"
        if path.exists():
            state = restore(policy, path)
            if (state["identity"], state["arm"], state["seed"]) != (
                manifest["identity"],
                arm,
                seed,
            ):
                raise ValueError("run identity mismatch")
        else:
            state = dict(
                identity=manifest["identity"],
                arm=arm,
                seed=seed,
                episodes=0,
                physical_ticks=0,
                updates=0,
                elapsed_seconds=0.0,
                early=None,
            )
            save(policy, path, state)
        reconcile(directory, state, manifest["early_ticks"])
        if (
            not state["episodes"]
            <= target
            <= min(manifest["episodes"], (state["episodes"] // 10 + 1) * 10)
        ):
            raise ValueError("target must stay within the next ten-episode block")

        def progress(extra=None):
            write(
                directory / "progress.json",
                {
                    **{k: v for k, v in state.items() if k != "last_episode"},
                    "pid": os.getpid(),
                    "updated_unix": time.time(),
                    **(extra or {}),
                },
            )

        for index in range(state["episodes"], target):
            start = time.monotonic()
            progress({"phase": "rollout"})
            world_seed = seed_for(seed, manifest["world_domain"], "train", index)
            report, rewards, critic = episode(
                factory,
                workload,
                options,
                policy,
                world_seed,
                CONDITIONS[index % 4],
                manifest["drain"],
                on_progress=lambda x: progress({"phase": "rollout", **x}),
            )
            progress({"phase": "ppo_update", "episode_tick": report["tick"]})
            losses = policy.update(
                [r / manifest["reward_divisor"] for r in rewards],
                terminal=report["status"] == "completed",
                final_critic=critic,
            )
            state = {k: v for k, v in state.items() if k != "last_episode"}
            state.update(
                episodes=index + 1,
                physical_ticks=state["physical_ticks"] + report["tick"],
                updates=policy.updates,
                elapsed_seconds=state["elapsed_seconds"] + time.monotonic() - start,
            )
            if (
                state["early"] is None
                and state["physical_ticks"] >= manifest["early_ticks"]
            ):
                state["early"] = {
                    k: state[k] for k in ("episodes", "physical_ticks", "updates")
                }
                state["early"].update(
                    threshold=manifest["early_ticks"],
                    overshoot=state["physical_ticks"] - manifest["early_ticks"],
                )
            state["last_episode"] = {
                **compact(report),
                **state,
                "losses": losses,
                "updated_unix": time.time(),
            }
            save(policy, path, state)
            reconcile(directory, state, manifest["early_ticks"])
            progress({"phase": "episode_saved"})
        # Engineering milestone 20 is deliberately outside the production retain schedule.
        if manifest["evidence"] == "engineering-probe" and state["episodes"] == 20:
            from smartsom.experiments.social_checkpoint import immutable_copy

            immutable_copy(path, directory / "ep-0020.pt")
        write(
            directory / f"chunk-{target:04d}.json",
            {k: v for k, v in state.items() if k != "last_episode"},
        )
        progress({"phase": "chunk_complete"})


class Rule:
    """Keep the established rule ranking; random samples the full legal action set."""

    def __init__(self, method):
        from smartsom.algorithms.base_rules import BaseRule

        self.base = BaseRule(method, 0)
        self.method = method

    def choose(self, request, critic=None):
        if self.method == "rule":
            candidates = tuple(
                c for c in request.candidates if c.action != ("EMPTY_WAIT",)
            )
            if not candidates:
                return ("EMPTY_WAIT",)
            request = replace(request, candidates=candidates)
        return self.base.choose(request, critic)


def evaluate(root, task):
    from smartsom.experiments.base_diagnostic import episode
    from smartsom.learning.base_ppo import BasePPO

    directory = root / "evaluation" / task["id"]
    with worker_guard(root, directory):
        manifest = verify(root)
        factory, workload, options = inputs(manifest, task["arm"])
        if task["method"] == "ppo":
            if digest(Path(task["checkpoint"])) != task["sha256"]:
                raise ValueError("evaluation checkpoint changed")
            policy = BasePPO(
                options, task["seed"], training=False, selection=task["decoder"]
            )
            policy.load(task["checkpoint"])
        else:
            policy = Rule(task["method"])
        rows = []
        for world in range(task["worlds"]):
            for repeat in range(task["repeats"]):
                path = directory / f"world-{world:02d}-{repeat}.json"
                if path.exists():
                    row = read(path)
                    if row["task"] != task:
                        raise ValueError("evaluation receipt identity mismatch")
                    rows.append(row)
                    continue
                scenario_seed = seed_for(
                    7101, manifest["world_domain"], task["split"], world
                )
                action_seed = seed_for(
                    8101, manifest["action_domain"], task["split"], world, repeat
                )
                if task["method"] == "ppo":
                    policy.generator.manual_seed(action_seed)
                else:
                    policy.base.rng.seed(action_seed)
                result, _, _ = episode(
                    factory,
                    workload,
                    options,
                    policy,
                    scenario_seed,
                    CONDITIONS[world % 4],
                    manifest["drain"],
                )
                row = {
                    **compact(result),
                    "world": world,
                    "repeat": repeat,
                    "action_seed": action_seed,
                    "task": task,
                }
                write(path, row)
                rows.append(row)
                write(
                    directory / "progress.json",
                    {
                        "done": len(rows),
                        "pid": os.getpid(),
                        "updated_unix": time.time(),
                    },
                )
        n = task["worlds"] * task["repeats"]
        drained = sum(r["unfinished"] == 0 for r in rows)
        write(
            directory / "summary.json",
            {
                "task": task,
                "actual": len(rows),
                "expected": n,
                "drained": drained,
                "delivery_fraction": sum(
                    r["completed"] / r["total_orders"] for r in rows
                )
                / n,
                "full_panel_mean_J": sum(r["joint_cost"] for r in rows) / n
                if drained == n
                else None,
                "finished_unix": time.time(),
            },
        )


def environment():
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    init = sub.add_parser("init")
    init.add_argument("--engineering", action="store_true")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--target", type=int, required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--task", type=Path, required=True)
    coordinator = sub.add_parser("coordinate")
    coordinator.add_argument("--legacy-root", type=Path)
    sub.add_parser("report")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        initialize(root, engineering=args.engineering)
    elif args.mode in ("train", "eval"):
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if args.mode == "train":
            train(root, args.arm, args.seed, args.target)
        else:
            evaluate(root, read(args.task))
    else:
        from smartsom.experiments.social_schedule import coordinate, report

        if args.mode == "coordinate":
            coordinate(root, args.legacy_root)
        else:
            report(root)


if __name__ == "__main__":
    main()
