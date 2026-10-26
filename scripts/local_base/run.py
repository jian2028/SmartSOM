"""Local reconstructed Base development experiment. No remote-source equivalence claim."""

import argparse
import csv
import fcntl
import hashlib
import html
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("normal", "low", "medium", "high")
ARMS = {"A": (True, False), "B": (False, False), "C": (True, True), "D": (False, True)}
MILESTONES = (0, 100, 200, 300, 500, 750, 1000)
MOVER_HASH = "57002337bd5510fc0e95028fd4d9aa1738d6cdd7ca2acbb2ea0b6c7fc677455b"


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(arm):
    from smartsom.config.base_diagnostic import load_inputs

    _, factory, workload, options, _ = load_inputs(
        ROOT / "configs/base-diagnostic/target.yaml"
    )
    stop, wait = ARMS[arm]
    return (
        factory,
        workload,
        options.model_copy(
            update={
                "reservation_stop": stop,
                "empty_wait": wait,
                "reconstructed_inputs": True,
            }
        ),
    )


def verify(root):
    manifest = read(root / "manifest.json")
    for name, sha in manifest["source_hashes"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"frozen source changed: {name}")
    import smartsom

    if not Path(smartsom.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(f"wrong source import: {smartsom.__file__}")
    return manifest


def seed_for(*args):
    from smartsom.config.base_diagnostic import seed_for as generate

    return generate(*args)


class Rule:
    """Same rule ranking; active WAIT permitted only when no source is eligible."""

    def __init__(self, method, seed):
        from smartsom.algorithms.base_rules import BaseRule

        self.base = BaseRule(method, seed)
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


def checkpoint(policy, path, state):
    import torch

    scratch = path.parent / f"checkpoint-{os.getpid()}.tmp.pt"
    policy.save(scratch)
    data = torch.load(scratch, map_location="cpu", weights_only=False)
    data["local_state"] = state
    data["torch_rng"] = torch.get_rng_state()
    torch.save(data, scratch)
    scratch.replace(path)


def restore(policy, path):
    import torch

    policy.load(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(data["torch_rng"])
    return data["local_state"]


def compact(report):
    return {
        k: v
        for k, v in report.items()
        if k not in ("final_traffic_state", "final_station_state", "horizon_checks")
    }


def train(root, arm, seed, target):
    from smartsom.experiments.base_diagnostic import episode
    from smartsom.learning.base_ppo import BasePPO

    manifest = verify(root)
    factory, workload, options = inputs(arm)
    directory = root / "train" / f"{arm}-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = BasePPO(options, seed)
    state = {
        "episodes": 0,
        "physical_ticks": 0,
        "updates": 0,
        "arm": arm,
        "seed": seed,
        "run_identity": manifest["identity"],
    }
    path = directory / "last.pt"
    if path.exists():
        state = restore(policy, path)
        last_row = state.pop("last_episode", None)
        if last_row:
            append(directory / "episodes.jsonl", last_row)
        assert state["run_identity"] == manifest["identity"]
    else:
        checkpoint(policy, path, state)
        shutil.copyfile(path, directory / "ep-0000.pt")

    def progress(extra=None):
        write(
            directory / "progress.json",
            {
                **state,
                "pid": os.getpid(),
                "updated_unix": time.time(),
                "source": str(ROOT),
                "phase": "rollout",
                **(extra or {}),
            },
        )

    progress()
    for index in range(state["episodes"], target):
        if time.time() >= manifest["training_cutoff_unix"]:
            break
        start = time.monotonic()
        scenario_seed = seed_for(seed, "mac-long-v1", "train", index)
        condition = CONDITIONS[index % 4]
        report, rewards, critic = episode(
            factory,
            workload,
            options,
            policy,
            scenario_seed,
            condition,
            2000,
            on_progress=progress,
            wall_deadline=time.monotonic()
            + max(0, manifest["training_cutoff_unix"] - time.time()),
        )
        progress({"phase": "ppo_update", "episode_tick": report["tick"]})
        losses = policy.update(
            [r / 1000.0 for r in rewards],
            terminal=report["status"] == "completed",
            final_critic=critic,
        )
        state = {
            **state,
            "episodes": index + 1,
            "physical_ticks": state["physical_ticks"] + report["tick"],
            "updates": policy.updates,
        }
        row = {
            **compact(report),
            **state,
            "elapsed_seconds": time.monotonic() - start,
            "losses": losses,
            "updated_unix": time.time(),
        }
        # The embedded episode row is authoritative if interruption falls between
        # the atomic checkpoint and the append-only ledger write.
        checkpoint(policy, path, {**state, "last_episode": row})
        append(directory / "episodes.jsonl", row)
        if state["episodes"] % 50 == 0:
            shutil.copyfile(path, directory / f"ep-{state['episodes']:04d}.pt")
        progress({"phase": "episode_saved"})
    write(
        directory / f"chunk-{target:04d}.json", {**state, "finished_unix": time.time()}
    )
    progress({"phase": "chunk_complete"})


def evaluate(root, task):
    from smartsom.experiments.base_diagnostic import episode
    from smartsom.learning.base_ppo import BasePPO

    manifest = verify(root)
    factory, workload, options = inputs(task["arm"])
    directory = root / "evaluation" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    method = task.get("method", "ppo")
    if method == "ppo":
        policy = BasePPO(
            options, task["seed"], training=False, selection=task["decoder"]
        )
        policy.load(Path(task["checkpoint"]))
    else:
        policy = Rule(method, 0)
    rows = []
    for world in range(task["worlds"]):
        for repeat in range(task["repeats"]):
            result_path = directory / f"world-{world:02d}-{repeat}.json"
            if result_path.exists():
                rows.append(read(result_path))
                continue
            if time.time() >= manifest["deadline_unix"]:
                raise TimeoutError("evaluation deadline")
            scenario_seed = seed_for(7101, "mac-long-v1", task["split"], world)
            action_seed = seed_for(8101, task["split"], world, repeat)
            if method == "ppo":
                policy.generator.manual_seed(action_seed)
            else:
                policy.base.rng.seed(action_seed)

            def progress(extra):
                write(
                    directory / "progress.json",
                    {
                        "pid": os.getpid(),
                        "updated_unix": time.time(),
                        "world": world,
                        "repeat": repeat,
                        "done": len(rows),
                        **extra,
                    },
                )

            result, _, _ = episode(
                factory,
                workload,
                options,
                policy,
                scenario_seed,
                CONDITIONS[world % 4],
                2000,
                on_progress=progress,
                wall_deadline=time.monotonic()
                + max(0, manifest["deadline_unix"] - time.time()),
            )
            row = {**compact(result), "world": world, "repeat": repeat, "task": task}
            write(result_path, row)
            rows.append(row)
    n = len(rows)
    complete = [r for r in rows if r["unfinished"] == 0]
    write(
        directory / "summary.json",
        {
            "task": task,
            "expected": task["worlds"] * task["repeats"],
            "actual": n,
            "drained": len(complete),
            "drained_rate": len(complete) / n,
            "delivery_fraction": sum(r["completed"] / r["total_orders"] for r in rows)
            / n,
            "full_panel_mean_J": sum(r["joint_cost"] for r in rows) / n
            if len(complete) == n
            else None,
            "successful_subset_mean_J": sum(r["joint_cost"] for r in complete)
            / len(complete)
            if complete
            else None,
            "finished_unix": time.time(),
        },
    )


def benchmark(root, arm, seed, label):
    from smartsom.experiments.base_diagnostic import episode
    from smartsom.learning.base_ppo import BasePPO

    verify(root)
    factory, workload, options = inputs(arm)
    policy = BasePPO(options, seed)
    start = time.monotonic()
    result, rewards, critic = episode(
        factory,
        workload,
        options,
        policy,
        seed_for(seed, "perf"),
        "normal",
        2000,
        max_ticks=1024,
    )
    policy.update([r / 1000 for r in rewards], terminal=False, final_critic=critic)
    write(
        root / "performance" / label / f"{arm}-{seed}.json",
        {
            "ticks": result["tick"],
            "updates": policy.updates,
            "seconds": time.monotonic() - start,
            "pid": os.getpid(),
            "arm": arm,
            "seed": seed,
        },
    )


def command(root, *args):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        *map(str, args),
    ]


def environment():
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }


def launch(root, task):
    log = root / "logs" / f"{task['id']}.log"
    log.parent.mkdir(exist_ok=True)
    if task["kind"] == "train":
        args = [
            "train",
            "--arm",
            task["arm"],
            "--seed",
            task["seed"],
            "--target",
            task["target"],
        ]
    else:
        path = root / "tasks" / f"{task['id']}.json"
        write(path, task)
        args = ["eval", "--task", path]
    with log.open("a") as f:
        process = subprocess.Popen(
            command(root, *args),
            cwd=ROOT,
            env=environment(),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    append(
        root / "experiment-ledger.jsonl",
        {
            "event": "launched",
            "pid": process.pid,
            "time": time.time(),
            "task": task,
            "command": process.args,
        },
    )
    return process


def resource_sample():
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    children = psutil.Process().children(recursive=True)
    rss = 0
    for child in children:
        try:
            rss += child.memory_info().rss
        except psutil.Error:
            pass
    return {
        "time": time.time(),
        "cpu_percent": psutil.cpu_percent(),
        "available_gib": memory.available / 2**30,
        "memory_percent": memory.percent,
        "swap_used_gib": swap.used / 2**30,
        "child_rss_gib": rss / 2**30,
        "children": len(children),
    }


def perf_pool(root, workers, label):
    start = time.monotonic()

    def one(index):
        arm, seed = list(ARMS)[index % 4], 9100 + index
        log = root / "performance" / label / f"{arm}-{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            result = subprocess.run(
                command(root, "bench", "--arm", arm, "--seed", seed, "--label", label),
                cwd=ROOT,
                env=environment(),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"benchmark failed: {log}")

    stop = threading.Event()

    def sample():
        while not stop.wait(2):
            append(root / "performance" / label / "resources.jsonl", resource_sample())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, range(48)))
    finally:
        stop.set()
        sampler.join()
    elapsed = time.monotonic() - start
    rows = [read(p) for p in (root / "performance" / label).glob("*.json")]
    summary = {
        "workers": workers,
        "tasks": len(rows),
        "seconds": elapsed,
        "ticks_per_second": sum(r["ticks"] for r in rows) / elapsed,
        "updates_per_second": sum(r["updates"] for r in rows) / elapsed,
        "finished_unix": time.time(),
        "scope": "1024-tick target rollouts plus actual PPO updates",
    }
    write(root / "performance" / f"{label}.json", summary)
    return summary


def eval_task(arm, seed, checkpoint_path, ep, decoder, split, worlds):
    return {
        "kind": "eval",
        "id": f"{split}-{arm}-{seed}-ep{ep}-{decoder}",
        "arm": arm,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "episode": ep,
        "method": "ppo",
        "decoder": decoder,
        "split": split,
        "worlds": worlds,
        "repeats": 2 if decoder == "sample" else 1,
    }


def curve_rows(root):
    rows = []
    for path in sorted((root / "train").glob("*/episodes.jsonl")):
        accepted = {}
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                accepted[row["episodes"]] = row
            except json.JSONDecodeError:
                continue
        rows.extend(accepted.values())
    return rows


def render(root, status):
    rows = curve_rows(root)
    keys = [
        "arm",
        "seed",
        "episodes",
        "physical_ticks",
        "updates",
        "unfinished",
        "completed",
        "total_orders",
        "observed_return",
        "joint_cost",
        "mean_tardiness",
        "elapsed_seconds",
    ]
    with (root / "learning-curves.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    panels = []
    colors = {"A": "#2563eb", "B": "#dc2626", "C": "#16a34a", "D": "#9333ea"}
    for seed in read(root / "manifest.json")["seeds"]:
        lines = []
        for arm in ARMS:
            subset = sorted(
                (r for r in rows if r["seed"] == seed and r["arm"] == arm),
                key=lambda r: r["episodes"],
            )
            # Delivery fraction keeps truncated episodes visible; complete J is separate in CSV.
            points = " ".join(
                f"{40 + r['episodes'] * 0.7:.1f},{220 - 180 * r['completed'] / r['total_orders']:.1f}"
                for r in subset
            )
            lines.append(
                f'<polyline fill="none" stroke="{colors[arm]}" stroke-width="2" points="{points}"/>'
            )
        panels.append(
            f'<h3>seed {seed}：订单交付比例（所有 episode，非整批排空率）</h3><svg viewBox="0 0 770 250"><path d="M40 20V220H740" fill="none" stroke="gray"/><text x="5" y="35">100%</text><text x="10" y="225">0%</text><text x="40" y="242">0</text><text x="690" y="242">1000 episode</text>{"".join(lines)}</svg>'
        )
    body = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
        "<title>Mac Base 长期诊断</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#f8fafc;color:#182230}svg{width:100%;background:white}pre{white-space:pre-wrap}</style>"
        "<h1>Mac 本地重建版 Base 长期诊断</h1><p>A 蓝：STOP / 无WAIT；B 红：无STOP / 无WAIT；C 绿：STOP / WAIT；D 紫：无STOP / WAIT。</p>"
        "<p>此为开发实验，非远端冻结版本精确复现。曲线为原始逐 episode 数据，需结合独立验证，不能凭训练回报判断收敛。</p>"
        f"<pre>{html.escape(json.dumps(status, ensure_ascii=False, indent=2))}</pre>"
        + "".join(panels)
    )
    (root / "dashboard.html").write_text(body)


def coordinate(root):
    manifest = verify(root)
    lock = (root / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    live, done, failed = {}, set(), set()
    for path in (root / "evaluation").glob("*/summary.json"):
        done.add(path.parent.name)
    for path in (root / "train").glob("*/chunk-*.json"):
        row = read(path)
        done.add(f"train-{row['arm']}-{row['seed']}-{int(path.stem.split('-')[1])}")
    ledger = root / "experiment-ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "exit" and row["code"]:
                failed.add(row["task"]["id"])
    final_scheduled = False
    last_render = 0
    while True:
        for key, (process, task) in list(live.items()):
            code = process.poll()
            if code is None:
                continue
            append(
                ledger,
                {
                    "event": "exit",
                    "pid": process.pid,
                    "code": code,
                    "task": task,
                    "time": time.time(),
                },
            )
            (failed if code else done).add(key)
            del live[key]
        now = time.time()
        control = read(root / "control.json")
        cap = min(18, max(1, control["workers"]))
        pending_train, pending_eval = [], []
        progress = {}
        train_running = {
            f"{t['arm']}-{t['seed']}" for _, t in live.values() if t["kind"] == "train"
        }
        for arm in ARMS:
            for seed in manifest["seeds"]:
                name = f"{arm}-{seed}"
                directory = root / "train" / name
                p = read(
                    directory / "progress.json",
                    {"episodes": 0, "physical_ticks": 0, "updates": 0},
                )
                progress[name] = p
                ep = p["episodes"]
                if (
                    now < manifest["training_cutoff_unix"]
                    and ep < 1000
                    and name not in train_running
                ):
                    target = min(1000, (ep // 10 + 1) * 10)
                    task = {
                        "kind": "train",
                        "id": f"train-{name}-{target}",
                        "arm": arm,
                        "seed": seed,
                        "target": target,
                    }
                    if task["id"] not in failed:
                        pending_train.append((ep, task))
                for checkpoint_ep in MILESTONES:
                    path = directory / f"ep-{checkpoint_ep:04d}.pt"
                    if path.exists():
                        for decoder in ("sample", "argmax"):
                            pending_eval.append(
                                eval_task(
                                    arm,
                                    seed,
                                    path,
                                    checkpoint_ep,
                                    decoder,
                                    "validation",
                                    8,
                                )
                            )
        for name in progress:
            arm, seed_string = name.split("-")
            path = root / "train" / name / "ep-0000.pt"
            if path.exists():
                for decoder in ("sample", "argmax"):
                    pending_eval.append(
                        eval_task(arm, int(seed_string), path, 0, decoder, "test", 32)
                    )
        # Baselines have each arm's action space; no across-arm result reuse.
        for arm in ARMS:
            for method in ("rule", "random"):
                pending_eval.append(
                    {
                        "kind": "eval",
                        "id": f"test-{arm}-{method}",
                        "arm": arm,
                        "method": method,
                        "split": "test",
                        "worlds": 32,
                        "repeats": 2 if method == "random" else 1,
                    }
                )
        training_done = (
            all(p["episodes"] >= 1000 for p in progress.values())
            or now >= manifest["training_cutoff_unix"]
        )
        validation_waiting = [
            t
            for t in pending_eval
            if t["split"] == "validation" and t["id"] not in done | failed
        ]
        if training_done and not train_running and not validation_waiting:
            selections = {}
            for name in progress:
                arm, seed_string = name.split("-")
                seed = int(seed_string)
                candidates = []
                for ep in MILESTONES:
                    if ep == 0:
                        continue
                    path = (
                        root
                        / "evaluation"
                        / f"validation-{name}-ep{ep}-sample"
                        / "summary.json"
                    )
                    if path.exists():
                        r = read(path)
                        if r["actual"] == r["expected"]:
                            score = (
                                r["drained_rate"],
                                r["delivery_fraction"],
                                -(
                                    r["full_panel_mean_J"]
                                    if r["full_panel_mean_J"] is not None
                                    else 1e30
                                ),
                                -ep,
                            )
                            candidates.append((score, ep))
                if candidates:
                    ep = max(candidates)[1]
                    selections[name] = ep
                    path = root / "train" / name / f"ep-{ep:04d}.pt"
                    for decoder in ("sample", "argmax"):
                        pending_eval.append(
                            eval_task(arm, seed, path, ep, decoder, "test", 32)
                        )
                else:
                    selections[name] = None
            write(root / "selected-checkpoints.json", selections)
            final_scheduled = True
        pending_eval = [
            t for t in pending_eval if t["id"] not in done | failed | live.keys()
        ]
        pending_train.sort(key=lambda pair: (pair[0], pair[1]["seed"], pair[1]["arm"]))
        # Train chunks balance all arms/seeds; evaluation borrows only spare slots
        # while at least cap-2 train slots remain occupied or queued.
        while (
            len(live) < cap
            and now < manifest["deadline_unix"]
            and not control.get("pause_admission", False)
        ):
            live_train = sum(t["kind"] == "train" for _, t in live.values())
            if pending_train and (
                live_train < max(1, cap - control.get("evaluation_slots", 2))
                or not pending_eval
            ):
                _, task = pending_train.pop(0)
            elif pending_eval:
                task = pending_eval.pop(0)
            elif pending_train:
                _, task = pending_train.pop(0)
            else:
                break
            live[task["id"]] = (launch(root, task), task)
        status = {
            "updated_unix": now,
            "pid": os.getpid(),
            "workers_limit": cap,
            "running": [{"pid": p.pid, **t} for p, t in live.values()],
            "train_progress": progress,
            "finished_tasks": len(done),
            "failed_tasks": sorted(failed),
            "deadline_unix": manifest["deadline_unix"],
            "final_scheduled": final_scheduled,
        }
        resources = resource_sample()
        status["resources"] = resources
        append(root / "resources.jsonl", resources)
        # Admission reduction does not kill running scientific work.
        if resources["available_gib"] < 8 and cap > 4:
            control["workers"] = max(4, cap - 2)
            write(root / "control.json", control)
            append(
                ledger,
                {
                    "event": "memory_admission_backoff",
                    "time": now,
                    "new_workers": control["workers"],
                },
            )
        write(root / "coordinator-status.json", status)
        if now - last_render > 60:
            render(root, status)
            last_render = now
        if now >= manifest["deadline_unix"]:
            # Stop only our registered child processes, preserve their last atomic
            # checkpoints and all partial evidence. No kill-by-name or remote action.
            for process, task in live.values():
                process.terminate()
                append(
                    ledger,
                    {
                        "event": "deadline_termination",
                        "pid": process.pid,
                        "task": task,
                        "time": now,
                    },
                )
            for process, _ in live.values():
                process.wait()
            break
        if final_scheduled and not live and not pending_train and not pending_eval:
            break
        time.sleep(5)
    status["status"] = (
        "finished" if time.time() < manifest["deadline_unix"] else "deadline"
    )
    status["remaining_at_exit"] = status["running"]
    status["running"] = []
    write(root / "coordinator-status.json", status)
    write(
        root / "exit-audit.json",
        {
            "pid": os.getpid(),
            "time": time.time(),
            "children_alive": [],
            "failed": sorted(failed),
        },
    )
    render(root, status)
    summaries = [read(p) for p in sorted((root / "evaluation").glob("*/summary.json"))]
    write(
        root / "comparison.json",
        {
            "status": status,
            "panels": summaries,
            "warning": "successful-subset J is descriptive only; never rank different success subsets",
        },
    )
    lines = [
        "# Mac 本地重建版 Base 长期诊断",
        "",
        "这是独立开发实验，不是远端冻结版本的精确复现。",
        "原奖励、Mover、320订单场景和排空2000不变。未排空的完整J不报告。",
        "",
        "## 训练实际完成范围",
        "",
        "|运行|episodes|物理tick|updates|",
        "|---|---:|---:|---:|",
    ]
    for name, row in status["train_progress"].items():
        lines.append(
            f"|{name}|{row['episodes']}|{row['physical_ticks']}|{row['updates']}|"
        )
    lines += [
        "",
        "## 已完成评价面板（缺失面板不是零分）",
        "",
        "|面板|已完成/预期|整批排空|平均交付比例|全量完整J|",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"|{r['task']['id']}|{r['actual']}/{r['expected']}|{r['drained_rate']:.3f}|{r['delivery_fraction']:.3f}|{r['full_panel_mean_J']}|"
        )
    lines += [
        "",
        "## 限制",
        "",
        "训练发生截断时，完整平均延期与完整J均不成立。单次轨迹不是收敛证据。",
        "本地重建未复制远端D3角色替换诊断；未增加奖励、Mover或关系编码修复。",
        "进程退出、失败任务与未完成训练见 exit-audit.json 和 coordinator-status.json。",
        "No-STOP为协议消融，不能直接认定是学习收益；需同动作空间rule/random及初始神经策略比较。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def initialize(root):
    if (root / "manifest.json").exists():
        raise RuntimeError("existing run; do not reinitialize")
    root.mkdir(parents=True, exist_ok=True)
    files = list((ROOT / "src/smartsom").rglob("*.py")) + list(
        (ROOT / "configs/base-diagnostic").glob("*.yaml")
    )
    files += list((ROOT / "scripts/local_base").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    assert hashes["src/smartsom/algorithms/base_rules.py"] == MOVER_HASH
    import psutil
    import torch

    now = time.time()
    manifest = {
        "identity": root.name,
        "kind": "local reconstructed development experiment",
        "source": str(ROOT),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": hashes,
        "exact_remote_replication": False,
        "seeds": [3101, 3102, 3103],
        "arms": ARMS,
        "orders": 320,
        "episodes_per_run": 1000,
        "started_unix": now,
        "training_cutoff_unix": 1793129900.769166,
        "deadline_unix": 1793144300.769166,
        "reward_target_divisor": 1000,
        "drain_ticks": 2000,
        "validation_worlds": 8,
        "test_worlds": 32,
        "milestones": MILESTONES,
        "checkpoint_every": 50,
        "atomic_resume_every": 1,
        "initial_workers": 8,
        "max_workers": 18,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "psutil": psutil.__version__,
        "mover_sha256": MOVER_HASH,
        "selection": "sample validation: drained fraction, delivery fraction, full-panel J when all drained, earlier episode tie-break",
        "differences": [
            "reconstructed input slots documented in docs/local-base-reconstruction.md",
            "no remote patch or checkpoints available; new paired seeds",
            "WAIT head in all five groups, unused outside dispatcher",
            "same original remote hard deadline, not a new 36-hour budget",
        ],
    }
    write(root / "manifest.json", manifest)
    write(
        root / "control.json",
        {"workers": 8, "evaluation_slots": 0, "pause_admission": False},
    )
    (root / "source.patch").write_text(
        subprocess.check_output(["git", "diff"], cwd=ROOT, text=True)
    )
    for name in hashes:
        destination = root / "frozen-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for arm in ARMS:
        _, _, options = inputs(arm)
        write(
            root / "configuration" / f"{arm}.json",
            options.model_dump(mode="json", by_alias=True),
        )
    print(
        json.dumps(
            {
                "root": str(root),
                "manifest": manifest["identity"],
                "sources": len(hashes),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "mode", choices=["init", "train", "eval", "bench", "perf", "coordinate"]
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target", type=int)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--label", default="c8")
    args = parser.parse_args()
    if args.mode in ("train", "eval", "bench"):
        import torch

        torch.set_num_interop_threads(1)
    if args.mode == "init":
        initialize(args.root)
    elif args.mode == "train":
        train(args.root, args.arm, args.seed, args.target)
    elif args.mode == "eval":
        evaluate(args.root, read(args.task))
    elif args.mode == "bench":
        benchmark(args.root, args.arm, args.seed, args.label)
    elif args.mode == "perf":
        print(json.dumps(perf_pool(args.root, args.workers, args.label)))
    else:
        coordinate(args.root)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
