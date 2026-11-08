"""One unattended check of a legacy official queue; schedule every 30 minutes.

Only official CLI commands mutate the queue. This script never edits its journal,
checkpoints or source, and never launches individual training workers.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(key, row):
    return hashlib.sha256(
        json.dumps(
            [
                key.split("::")[-1],
                *[row.get(k) for k in ("pid", "retry_attempt", "exit_code")],
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def live_processes(queue, tasks):
    """Conservative Linux liveness gate; official recovery rechecks ownership."""
    path = queue / "control/owner.json"
    owner = json.loads(path.read_text()) if path.exists() else {}
    records = ([owner["owner"]] if owner.get("owner") else []) + owner.get(
        "members", []
    )
    for record in records:
        path = Path("/proc") / str(record["pid"]) / "stat"
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if fields[19] == str(record["created"]) and fields[0] != "Z":
            return True
    # A live unregistered child must also block recovery.
    return any(
        row.get("pid") and (Path("/proc") / str(row["pid"])).exists()
        for row in tasks.values()
        if row.get("status") in {"running", "starting", "queued"}
    )


def check(config, state_path, call, alive=live_processes):
    state = json.loads(state_path.read_text()) if state_path.exists() else {"seen": []}
    if state.get("phase") == "followup":
        config = {**config, **config["followup"]}
    queue = Path(config["queue"])
    snapshot = call(["batch-status", str(queue)])
    tasks = {**snapshot.get("tasks", {}), **snapshot.get("additions", {})}
    # Explicit manual holds take precedence over unattended recovery.
    if (
        snapshot.get("paused")
        or snapshot.get("stopping")
        or snapshot.get("paused_tasks")
        or snapshot.get("stopped_tasks")
        or snapshot.get("removed_tasks")
    ):
        return {"action": "held", "status": snapshot.get("status")}
    failed = {key: row for key, row in tasks.items() if row.get("status") == "failed"}
    fresh = sorted(
        {fingerprint(key, row) for key, row in failed.items()} - set(state["seen"])
    )
    if fresh and not state.get("pending"):
        state["pending"] = {
            "failures": fresh,
            "limit": max(1, snapshot["limit"] - 1),
            "command_id": "watch-" + uuid.uuid4().hex,
        }
        save(state_path, state)
    if state.get("pending"):
        pending = state["pending"]
        receipt = call(
            [
                "batch-limit",
                str(queue),
                str(pending["limit"]),
                "--command-id",
                pending["command_id"],
            ]
        )
        if receipt.get("status") in {"rejected", "failed"}:
            raise RuntimeError(f"limit rejected: {receipt}")
        # The idempotent receipt survives a crash between submission and save.
        state["seen"] = sorted(set(state["seen"]) | set(pending["failures"]))
        state["last_limit"] = pending["limit"]
        del state["pending"]
        save(state_path, state)
    actions = []
    living = alive(queue, tasks)
    if living:
        state["needs_recovery"] = False
    followup = config.get("followup") if not state.get("phase") else None
    originals_finished = all(
        any(
            key.endswith("--" + entry + "--seed-101")
            and row.get("status") == "completed"
            for key, row in tasks.items()
        )
        for entry in config["entries"]
    )
    if followup and originals_finished and not living:
        target = Path(followup["queue"])
        owner_state = json.loads((target / "batch.json").read_text())
        admission = json.loads((target / "control/admission.json").read_text())
        if any(row.get("status") == "failed" for row in owner_state["files"].values()):
            if admission.get("queue"):
                raise RuntimeError("external owner has not detached; retry next check")
            receipt = call(
                [
                    "batch-limit",
                    str(target),
                    str(state.get("last_limit", snapshot["limit"])),
                    "--command-id",
                    "watch-followup-limit",
                ]
            )
            if receipt.get("status") in {"rejected", "failed"}:
                raise RuntimeError(f"handoff limit rejected: {receipt}")
            state.update(phase="followup", needs_recovery=True)
            save(state_path, state)
            return check(config, state_path, call, alive)
    unfinished = [
        row
        for row in tasks.values()
        if row.get("status") in {"failed", "running", "queued", "starting"}
    ]
    if (unfinished or state.get("needs_recovery")) and not living:
        command = ["batch-run", config["source"], "--resume-missing"]
        for entry in config["entries"]:
            command += ["--id", entry]
        command += ["--output-root", str(queue), "--background"]
        call(command, parse=False)
        actions.append("official batch recovery submitted")
        state["needs_recovery"] = False
    elif failed:
        for key in failed:
            try:
                call(["batch-retry", str(queue), "--task", key], parse=False)
                actions.append({"task": key, "retry": "submitted"})
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                actions.append({"task": key, "retry": "blocked", "reason": str(exc)})
    state["last_checked_at"] = time.time()
    state["last_result"] = {
        "failures": len(failed),
        "new_failures": len(fresh),
        "limit": state.get("last_limit", snapshot["limit"]),
        "actions": actions,
    }
    save(state_path, state)
    return state["last_result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def call(arguments, parse=True):
            result = subprocess.run(
                [config["python"], "-m", "smartsom.experiments.cli", *arguments],
                cwd=config["runtime"],
                env={**os.environ, "PYTHONPATH": str(Path(config["runtime"]) / "src")},
                text=True,
                capture_output=True,
                timeout=600 if arguments[0] == "batch-run" else 60,
            )
            if result.returncode:
                raise RuntimeError(
                    f"{arguments[0]}: {result.stderr[-2000:]} {result.stdout[-2000:]}"
                )
            return json.loads(result.stdout) if parse else None

        print(
            json.dumps(check(config, args.state, call), ensure_ascii=False), flush=True
        )


if __name__ == "__main__":
    main()
