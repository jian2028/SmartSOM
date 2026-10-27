"""Single-controller fair blocks and conservative shared-host admission."""

import csv
import html
import io
import os
import subprocess
import sys
import time

from smartsom.experiments.social_checkpoint import atomic_bytes, digest, read, write
from smartsom.experiments.social_long import ROOT, environment, lock_file, verify


def snapshot(legacy_root=None):
    import psutil

    # A measured interval is required; never admit on an unprimed zero sample.
    cpu = psutil.cpu_percent(interval=0.3)
    scientific = []
    for process in psutil.process_iter(["pid", "ppid", "cmdline", "name", "status"]):
        try:
            info = process.info
            if info["status"] == psutil.STATUS_ZOMBIE or info["pid"] == os.getpid():
                continue
            command = " ".join(info["cmdline"] or [])
            name = info["name"] or ""
            if (
                ("python" in name.lower() or name.startswith("ray::"))
                and not ("social_long" in command and " coordinate" in command)
                and "supervisor.py" not in command
            ):
                scientific.append(
                    {
                        "pid": info["pid"],
                        "command": command,
                        "threads": process.num_threads(),
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    legacy_alive = False
    legacy_count = 0
    if legacy_root:
        for process in psutil.process_iter(["pid", "cmdline"]):
            command = " ".join(process.info["cmdline"] or [])
            if str(legacy_root) in command and "supervisor.py" in command:
                legacy_alive = True
        legacy_count = sum(str(legacy_root) in p["command"] for p in scientific)
    return {
        "unix": time.time(),
        "cpu_percent": cpu,
        "cpus": psutil.cpu_count(),
        "available_gib": psutil.virtual_memory().available / 2**30,
        "scientific": scientific,
        "legacy_alive": legacy_alive,
        "legacy_count": legacy_count,
        "reserved_idle_legacy": max(0, 24 - legacy_count) if legacy_alive else 0,
    }


def can_admit(sample, own_count, control):
    # The old scheduler retains all 24 promised slots. Latest user authorization
    # permits 28 combined slots; initial SL admission remains capped at four.
    return (
        not control.get("pause_admission", False)
        and own_count < min(4, control["workers"])
        and sample["available_gib"] >= 16
        and len(sample["scientific"]) + sample["reserved_idle_legacy"]
        < min(28, control["total_limit"])
        and sample["cpu_percent"] < 85
        and sample["cpus"] * (1 - sample["cpu_percent"] / 100) >= 2
    )


def evaluation_tasks(root, manifest, split):
    tasks = []
    labels = [("initial", "ep-0000.pt")]
    if split == "validation":
        labels += [(f"ep{e}", f"ep-{e:04d}.pt") for e in manifest["milestones"] if e]
        labels += [("early-160k", "early-160k.pt")]
    else:
        labels += [
            ("early-160k", "early-160k.pt"),
            (f"ep{manifest['episodes']}", f"ep-{manifest['episodes']:04d}.pt"),
        ]
    for arm in manifest["arms"]:
        for seed in manifest["seeds"]:
            directory = root / "train" / f"{arm}-{seed}"
            for label, filename in labels:
                path = directory / filename
                if not path.exists():
                    continue
                sha = digest(path)
                for decoder in ("sample", "argmax"):
                    tasks.append(
                        {
                            "kind": "eval",
                            "id": f"{split}-{arm}-{seed}-{label}-{decoder}",
                            "arm": arm,
                            "seed": seed,
                            "label": label,
                            "method": "ppo",
                            "checkpoint": str(path),
                            "sha256": sha,
                            "decoder": decoder,
                            "split": split,
                            "worlds": manifest[f"{split}_worlds"],
                            "repeats": 2 if decoder == "sample" else 1,
                        }
                    )
    # These controls ignore social data and all arms share exactly the same
    # protocol/options/masks. They are one shared control, never nine replications.
    for method in ("rule", "random"):
        tasks.append(
            {
                "kind": "eval",
                "id": f"{split}-shared-{method}",
                "arm": "base",
                "method": method,
                "split": split,
                "worlds": manifest[f"{split}_worlds"],
                "repeats": 2 if method == "random" else 1,
                "shared_across_arms": True,
            }
        )
    return tasks


def task_done(root, task):
    if task["kind"] == "train":
        path = (
            root
            / "train"
            / f"{task['arm']}-{task['seed']}"
            / f"chunk-{task['target']:04d}.json"
        )
        return path.exists() and read(path)["episodes"] == task["target"]
    path = root / "evaluation" / task["id"] / "summary.json"
    if not path.exists():
        return False
    result = read(path)
    if result["task"] != task or result["actual"] != result["expected"]:
        raise ValueError(f"invalid completed task: {task['id']}")
    return True


def queue(root, manifest):
    training, progress = [], {}
    for arm in manifest["arms"]:
        for seed in manifest["seeds"]:
            name = f"{arm}-{seed}"
            path = root / "train" / name / "state.json"
            state = (
                read(path)
                if path.exists()
                else {"episodes": 0, "physical_ticks": 0, "updates": 0}
            )
            progress[name] = state
            if state["episodes"] < manifest["episodes"]:
                target = min(manifest["episodes"], (state["episodes"] // 10 + 1) * 10)
                training.append(
                    {
                        "kind": "train",
                        "id": f"train-{name}-{target}",
                        "arm": arm,
                        "seed": seed,
                        "target": target,
                    }
                )
    training.sort(
        key=lambda t: (t["target"], t["seed"], manifest["arms"].index(t["arm"]))
    )
    validation = evaluation_tasks(root, manifest, "validation")
    waiting_validation = [t for t in validation if not task_done(root, t)]
    # Test is unavailable until ALL training and ALL required validation finishes.
    evaluation = waiting_validation
    if not training and not waiting_validation:
        expected = (
            len(manifest["arms"])
            * len(manifest["seeds"])
            * (len(manifest["milestones"]) + 1)
            * 2
            + 2
        )
        if len(validation) != expected:
            raise ValueError("missing mandatory validation checkpoint; refusing test")
        evaluation = [
            t
            for t in evaluation_tasks(root, manifest, "test")
            if not task_done(root, t)
        ]
    return training, evaluation, progress


def reuse(root, task):
    if task["method"] != "ppo":
        return False
    # Only labels of the SAME run/checkpoint/decoder/split may share evidence.
    keys = ("arm", "seed", "sha256", "decoder", "split", "worlds", "repeats", "method")
    for path in (root / "evaluation").glob("*/summary.json"):
        result = read(path)
        if all(result["task"].get(k) == task.get(k) for k in keys):
            if result["actual"] != result["expected"]:
                raise ValueError("incomplete reusable panel")
            write(
                root / "evaluation" / task["id"] / "summary.json",
                {**result, "task": task, "reused_from": result["task"]["id"]},
            )
            return True
    return False


def report(root):
    manifest = read(root / "manifest.json")
    training = []
    for path in sorted((root / "train").glob("*/episodes.jsonl")):
        # In-flight partial append is not used: the ledger is atomically replaced.
        training.extend(
            __import__("json").loads(line) for line in path.read_text().splitlines()
        )
    fields = [
        "arm",
        "seed",
        "episodes",
        "physical_ticks",
        "elapsed_seconds",
        "updates",
        "condition",
        "completed",
        "total_orders",
        "unfinished",
        "joint_cost",
        "observed_return",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(training)
    atomic_bytes(root / "learning-curves.csv", out.getvalue().encode())
    panels = [read(p) for p in sorted((root / "evaluation").glob("*/summary.json"))]
    out = io.StringIO()
    fields = [
        "id",
        "arm",
        "seed",
        "label",
        "split",
        "method",
        "decoder",
        "sha256",
        "actual",
        "expected",
        "drained",
        "delivery_fraction",
        "full_panel_mean_J",
        "reused_from",
    ]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(
        {**p["task"], **{k: v for k, v in p.items() if k != "task"}} for p in panels
    )
    atomic_bytes(root / "evaluation-panels.csv", out.getvalue().encode())
    lines = [
        "# SL frozen-feature development comparison",
        "",
        f"Source: {manifest['commit']}; evidence: {manifest['evidence']}.",
        "All 27 seeds/arms remain in the panel. Missing panels are not zero scores.",
        "Only fully drained panels have full J; no ranking of different successful subsets.",
        "Predeclared endpoints: early-160k and episode 1000; no test-based selection.",
        "",
        "|Run|Episode|Training tick|Updates|Elapsed seconds|",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in manifest["arms"]:
        for seed in manifest["seeds"]:
            rows = [r for r in training if r["arm"] == arm and r["seed"] == seed]
            r = (
                rows[-1]
                if rows
                else dict(episodes=0, physical_ticks=0, updates=0, elapsed_seconds=0)
            )
            lines.append(
                f"|{arm}-{seed}|{r['episodes']}|{r['physical_ticks']}|{r['updates']}|{r['elapsed_seconds']:.1f}|"
            )
    lines += [
        "",
        "|Evaluation|Completed/expected|Drained|Delivery fraction|Full-panel J|",
        "|---|---:|---:|---:|---:|",
    ]
    for p in panels:
        lines.append(
            f"|{p['task']['id']}|{p['actual']}/{p['expected']}|{p['drained']}|{p['delivery_fraction']:.4f}|{p['full_panel_mean_J']}|"
        )
    atomic_bytes(root / "report.md", ("\n".join(lines) + "\n").encode())
    charts = []
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be185d",
        "#64748b",
        "#111827",
    ]
    for seed in manifest["seeds"]:
        for axis in ("episodes", "physical_ticks", "elapsed_seconds"):
            subset = [r for r in training if r["seed"] == seed]
            xmax = max([r[axis] for r in subset] + [1])
            paths = []
            for arm, color in zip(manifest["arms"], colors, strict=True):
                points = " ".join(
                    f"{40 + 700 * r[axis] / xmax:.2f},{220 - 180 * r['completed'] / r['total_orders']:.2f}"
                    for r in subset
                    if r["arm"] == arm
                )
                paths.append(
                    f'<polyline fill="none" stroke="{color}" points="{points}"/>'
                )
            charts.append(
                f'<h3>Seed {seed}: delivery fraction / {axis} (0–{xmax:.0f})</h3><svg viewBox="0 0 770 250"><path d="M40 20V220H740" fill="none" stroke="gray"/>{"".join(paths)}</svg>'
            )
    legend = " · ".join(
        f'<span style="color:{c}">{a}</span>'
        for a, c in zip(manifest["arms"], colors, strict=True)
    )
    atomic_bytes(
        root / "dashboard.html",
        (
            '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>SL progress</title><style>body{font:15px system-ui;max-width:1000px;margin:30px auto}svg{width:100%}pre{white-space:pre-wrap}</style><h1>SL development comparison</h1>'
            + legend
            + "".join(charts)
            + "<pre>"
            + html.escape("\n".join(lines))
            + "</pre>"
        ).encode(),
    )


def coordinate(root, legacy_root=None):
    manifest = verify(root)
    with lock_file(root / "coordinator.lock") as lock:
        if (root / "failure.json").exists() or any((root / "failures").glob("*.json")):
            raise RuntimeError(
                "previous failure requires explicit diagnosis; no automatic retry"
            )
        live = {}
        last_report = 0
        try:
            while True:
                if any((root / "failures").glob("*.json")):
                    raise RuntimeError(
                        "worker failure receipt requires explicit diagnosis"
                    )
                for key, (process, task) in list(live.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del live[key]
                    if code or not task_done(root, task):
                        raise RuntimeError(
                            f"worker {key} failed: exit={code}; see logs/{key}.log"
                        )
                training, evaluation, progress = queue(root, manifest)
                running_names = {
                    f"{t['arm']}-{t['seed']}"
                    for _, t in live.values()
                    if t["kind"] == "train"
                }
                training = [
                    t
                    for t in training
                    if f"{t['arm']}-{t['seed']}" not in running_names
                ]
                evaluation = [t for t in evaluation if t["id"] not in live]
                control = read(root / "control.json")
                sample = snapshot(legacy_root)
                # The training wave cannot move beyond the least advanced run's block.
                # This also makes restart ordering independent of wall-clock speed.
                frontier = min(
                    (p["episodes"] // 10 + 1) * 10 for p in progress.values()
                )
                training = [t for t in training if t["target"] <= frontier]
                while can_admit(sample, len(live), control) and (
                    training or evaluation
                ):
                    task = training.pop(0) if training else evaluation.pop(0)
                    if task["kind"] == "eval" and reuse(root, task):
                        continue
                    args = [
                        sys.executable,
                        "-m",
                        "smartsom.experiments.social_long",
                        "--root",
                        str(root),
                    ]
                    if task["kind"] == "train":
                        args += [
                            "train",
                            "--arm",
                            task["arm"],
                            "--seed",
                            str(task["seed"]),
                            "--target",
                            str(task["target"]),
                        ]
                    else:
                        path = root / "tasks" / f"{task['id']}.json"
                        write(path, task)
                        args += ["eval", "--task", str(path)]
                    log = root / "logs" / f"{task['id']}.log"
                    log.parent.mkdir(exist_ok=True)
                    # Inherit the controller lock: orphan workers prevent a second
                    # controller until they finish their bounded block/panel.
                    with log.open("a") as output:
                        process = subprocess.Popen(
                            ["nice", "-n", "10", *args],
                            cwd=ROOT,
                            env=environment(),
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),),
                        )
                    live[task["id"]] = (process, task)
                    write(
                        root / "launches" / f"{task['id']}.json",
                        {
                            "pid": process.pid,
                            "command": process.args,
                            "task": task,
                            "unix": time.time(),
                            "threads": 1,
                        },
                    )
                    sample = snapshot(legacy_root)
                status = {
                    "pid": os.getpid(),
                    "unix": time.time(),
                    "status": "running" if live else "waiting",
                    "resources": sample,
                    "progress": progress,
                    "running": [{"pid": p.pid, **t} for p, t in live.values()],
                    "pending_train": len(training),
                    "pending_evaluation": len(evaluation),
                    "control": control,
                }
                write(root / "status.json", status)
                if time.monotonic() - last_report > 60:
                    report(root)
                    last_report = time.monotonic()
                if not live and not training and not evaluation:
                    write(root / "status.json", {**status, "status": "finished"})
                    report(root)
                    return
                time.sleep(5)
        except BaseException as error:
            write(
                root / "failure.json",
                {
                    "error": repr(error),
                    "unix": time.time(),
                    "running": [{"pid": p.pid, **t} for p, t in live.values()],
                },
            )
            # Do not kill scientific workers. Wait for their finite block to finish;
            # their inherited lock also protects a controller killed externally.
            for process, _ in live.values():
                process.wait()
            raise
