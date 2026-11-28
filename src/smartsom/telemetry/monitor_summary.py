"""Read-only unified host and batch summary export for terminal and external monitors."""

import hashlib
import json
import os
import subprocess
import time
from functools import cache
from pathlib import Path

from smartsom.config.launch import project_root
from smartsom.config.run_meta import (
    build_auto_summary,
    host_identity,
    placeholder_eta,
    resolved_meta,
)
from smartsom.experiments.runtime_control import QUEUED as RUNTIME_QUEUED
from smartsom.telemetry.runtime import FAILURES as RUNTIME_FAILURES
from smartsom.telemetry.runtime import FINAL as RUNTIME_FINAL

SUMMARY_SCHEMA = "smartsom.monitor-summary/v1"
EXCLUDED_PARTS = {
    "inputs",
    "source",
    "inventory",
    "tracking",
    "continuation",
    "checkpoints",
    "replay",
}
# Buckets partition every status: anything not named below counts as "other",
# so the bucket counts always sum to the total.
ACTIVE_STATUSES = {
    "running",
    "starting",
    "active",
    "draining",
    "pausing",
    "stop_requested",
    "stopping",
    "training",
    "evaluating",
    "calibrating",
    "smoke",
}
QUEUED_STATUSES = RUNTIME_QUEUED | {"prepared", "allocated"}
PAUSED_STATUSES = {"paused"}
FAILED_STATUSES = RUNTIME_FAILURES | {"force_stopped"}
COMPLETED_STATUSES = {"completed", "wall_budget_reached", "frozen", "recommended"}
REMOVED_STATUSES = {"removed", "removing"}
# Ended without failure before finishing all selected work.
STOPPED_STATUSES = (
    RUNTIME_FINAL - FAILED_STATUSES - COMPLETED_STATUSES - REMOVED_STATUSES
)
BUCKETS = (
    ("active", ACTIVE_STATUSES),
    ("queued", QUEUED_STATUSES),
    ("paused", PAUSED_STATUSES),
    ("completed", COMPLETED_STATUSES),
    ("failed", FAILED_STATUSES),
    ("stopped", STOPPED_STATUSES),
    ("removed", REMOVED_STATUSES),
)


def _bucket(status) -> str:
    for name, statuses in BUCKETS:
        if status in statuses:
            return name
    return "other"


def _bucket_counts(statuses=(), *, buckets=()) -> dict[str, int]:
    counts = dict.fromkeys([name for name, _ in BUCKETS] + ["other"], 0)
    for name in [*map(_bucket, statuses), *buckets]:
        counts[name] += 1
    return counts


def _record_bucket(record: dict) -> str:
    """A partial batch whose selected tasks all completed only defers later seeds."""
    progress = record.get("progress") or {}
    if (
        record["status"] == "partial"
        and progress.get("total_tasks")
        and progress.get("completed_tasks") == progress["total_tasks"]
    ):
        return "completed"
    return _bucket(record["status"])


def dead_driver_status(
    root: Path, process_table, recorded_host, journal_status, ledger_status
):
    """Return the status of a run whose recorded processes are confirmed gone.

    None means the run may still be live or liveness cannot be judged here.
    """
    from smartsom.experiments.control import (
        alive,
        read,
        session_context,
        session_members,
    )

    try:
        owner = read(root)
    except (OSError, ValueError):
        return None
    if not isinstance(owner, dict) or not isinstance(owner.get("owner"), dict):
        return None  # No owner record: liveness cannot be judged.
    context, here = owner["owner"].get("context"), session_context()
    if context not in (None, here):
        if (
            str(context).startswith("linux:")
            and str(context).rsplit(":", 1)[0] == str(here).rsplit(":", 1)[0]
        ):
            return None  # Same Linux boot, another PID namespace.
        if recorded_host != host_identity()["name"]:
            return None  # Another host; on the recorded host this means a reboot.
    try:
        table = process_table()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None  # The process table is unavailable on this platform.
    # Same predicate as the delete guard: recorded processes, owned sessions
    # and a background launcher that has not yet bound a new owner.
    identities = [owner["owner"], *(owner.get("members") or [])]
    launcher = _read_json(root / "control/background-launch.lock")
    if (
        any(isinstance(item, dict) and alive(item, table) for item in identities)
        or session_members(owner, table)
        or (launcher.get("pid") is not None and alive(launcher, table))
    ):
        return None
    if ledger_status in RUNTIME_FINAL:
        return ledger_status
    if owner.get("status") in RUNTIME_FINAL:
        return owner["status"]
    stopping = {"stop_requested", "stopping"}
    stop = _read_json(root / "control/stop.json")
    if (
        journal_status in stopping
        or owner.get("status") in stopping
        or (stop.get("id") is not None and stop.get("id") == owner.get("id"))
    ):
        return "stopped"  # An operator stop, not a crash.
    return "interrupted"  # SIGKILL, OOM or a reboot skipped the final write.


def _recorded_host(*documents):
    for document in documents:
        host = (document.get("facts") or {}).get("host")
        if isinstance(host, dict) and host.get("name"):
            return host["name"]
    return None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def host_resources(*, cpu_interval: float | None = 0.1) -> dict:
    """Collect host CPU and memory observations without requiring psutil."""
    info = {
        **host_identity(),
        "cpu_percent": None,
        "cpus_used": None,
        "cpus_total": os.cpu_count(),
        "memory_used_bytes": None,
        "memory_total_bytes": None,
        "memory_used_gib": None,
        "memory_total_gib": None,
        "memory_percent": None,
    }
    try:
        import psutil

        cpus = psutil.cpu_count(logical=True) or os.cpu_count()
        cpu_pct = psutil.cpu_percent(interval=cpu_interval)
        vm = psutil.virtual_memory()
        used_bytes = vm.total - vm.available
        info.update(
            cpu_percent=round(float(cpu_pct), 1),
            cpus_used=round(float(cpus * cpu_pct / 100.0), 2) if cpus else None,
            cpus_total=cpus,
            memory_used_bytes=int(used_bytes),
            memory_total_bytes=int(vm.total),
            memory_used_gib=round(used_bytes / 1024**3, 2),
            memory_total_gib=round(vm.total / 1024**3, 2),
            memory_percent=round(float(vm.percent), 1),
        )
    except Exception:
        pass
    return info


def discover_top_level_runs(roots=None) -> list[Path]:
    """Find run/batch directories, excluding nested child trials owned by a parent batch."""
    roots = roots or [project_root() / "runs"]
    found: set[Path] = set()
    for source in roots:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            continue
        for directory, children, files in os.walk(root):
            if {"plan.json", "run.json", "manifest.json"}.intersection(files):
                found.add(Path(directory))
                children.clear()
            else:
                children[:] = [
                    child for child in children if child not in EXCLUDED_PARTS
                ]
    ordered = sorted(found, key=lambda p: (len(p.parts), str(p)))
    top_level: list[Path] = []
    for candidate in ordered:
        if any(candidate.is_relative_to(parent) for parent in top_level):
            continue
        top_level.append(candidate)
    return top_level


def _best_mtime(root: Path, files: tuple[str, ...]) -> float | None:
    times = []
    for rel in files:
        path = root / rel
        if path.is_file():
            try:
                times.append(path.stat().st_mtime)
            except OSError:
                pass
    return max(times) if times else None


def _child_progress(root: Path, plan: dict, runtime: dict) -> list[dict]:
    rows = []
    files = plan.get("files") if isinstance(plan.get("files"), list) else []
    runtime_tasks = {
        **(runtime.get("tasks") or {}),
        **(runtime.get("additions") or {}),
    }
    seen_ids = set()
    for item in files:
        tid = item.get("id")
        child = root / item["child"] if item.get("child") else None
        prog = _read_json(child / "progress.json") if child else {}
        rt = runtime_tasks.get(tid, {})
        status = rt.get("status") or prog.get("status") or "unknown"
        rows.append({"id": tid, "status": status, "progress": prog})
        seen_ids.add(tid)
    for tid, job in (runtime.get("additions") or {}).items():
        if tid in seen_ids:
            continue
        child = Path(job["root"]) if job.get("root") else None
        prog = _read_json(child / "progress.json") if child else {}
        rows.append(
            {
                "id": tid,
                "status": job.get("status") or prog.get("status") or "unknown",
                "progress": prog,
            }
        )
    return rows


def _progress_fraction(progress, status):
    """Use committed work in its native unit, including early terminal outcomes."""
    for counter, budget in (
        ("episodes", "episode_budget"),
        ("physical_ticks", "tick_budget"),
        ("completed_cases", "total_cases"),
    ):
        done, total = progress.get(counter), progress.get(budget)
        if (
            isinstance(done, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            ceiling = 1.0 if status in COMPLETED_STATUSES else 0.99
            return min(ceiling, max(0.0, done / total))
    if (
        status in COMPLETED_STATUSES
        and progress.get("status", status) != "wall_budget_reached"
    ):
        return 1.0
    return None


def _aggregate_progress(
    root: Path,
    plan: dict,
    batch: dict,
    runtime: dict,
    progress_doc: dict,
    live_snapshot: dict,
    status: str,
    alive_driver: bool = True,
) -> dict:
    is_batch = plan.get("schema") == "smartsom.author-batch-plan/v1" or isinstance(
        plan.get("files"), list
    )
    if is_batch:
        children = _child_progress(root, plan, runtime)
        batch_files = batch.get("files") or {}
        if children:
            for child in children:
                if child["status"] == "unknown" and child["id"] in batch_files:
                    child["status"] = batch_files[child["id"]].get("status", "unknown")
        elif isinstance(live_snapshot.get("tasks"), list):
            children = [
                {
                    "id": row.get("id"),
                    "status": row.get("status", "unknown"),
                    "progress": {
                        **(row.get("values") or {}),
                        "episodes": row.get("completed")
                        if row.get("unit") == "episodes"
                        else (row.get("values") or {}).get("episodes"),
                        "episode_budget": row.get("total")
                        if row.get("unit") == "episodes"
                        else None,
                        "physical_ticks": (row.get("values") or {}).get(
                            "committed_physical_ticks"
                        ),
                    },
                }
                for row in live_snapshot["tasks"]
            ]

        if not alive_driver:
            # A dead driver leaves its last journal behind; nothing is running now.
            for child in children:
                if child["status"] in ACTIVE_STATUSES:
                    child["status"] = status
        deferred = sum(c["status"] == "deferred" for c in children)
        non_deferred = [c for c in children if c["status"] != "deferred"]
        buckets = _bucket_counts(c["status"] for c in non_deferred or children)
        total = len(non_deferred) or len(children)

        ep_counts, ep_budgets, fractions = [], [], []
        ticks_done, ticks_budget = 0, 0
        has_tick_budget = False
        for child in non_deferred:
            prog = child["progress"]
            ep = prog.get("episodes")
            ep_b = prog.get("episode_budget")
            if isinstance(ep_b, (int, float)) and ep_b > 0:
                ep_budgets.append(int(ep_b))
            if isinstance(ep, (int, float)):
                ep_counts.append(int(ep))
            fractions.append(_progress_fraction(prog, child["status"]) or 0.0)
            if isinstance(prog.get("physical_ticks"), (int, float)):
                ticks_done += int(prog["physical_ticks"])
            if (
                isinstance(prog.get("tick_budget"), (int, float))
                and prog["tick_budget"] > 0
            ):
                ticks_budget += int(prog["tick_budget"])
                has_tick_budget = True

        active_eps = [
            int(c["progress"]["episodes"])
            for c in non_deferred
            if c["status"] in ACTIVE_STATUSES
            and isinstance(c["progress"].get("episodes"), (int, float))
        ]
        display_eps = active_eps or ep_counts
        budget_set = set(ep_budgets)
        fraction = round(sum(fractions) / len(fractions), 4) if fractions else None
        return {
            **{f"{name}_tasks": count for name, count in buckets.items()},
            "deferred_tasks": deferred,
            "total_tasks": total,
            "fraction": fraction,
            "episodes": {
                "min": min(display_eps) if display_eps else None,
                "max": max(display_eps) if display_eps else None,
                "budget": next(iter(budget_set)) if len(budget_set) == 1 else None,
            },
            "physical_ticks": {
                "completed": ticks_done if ticks_done else None,
                "budget": ticks_budget if has_tick_budget else None,
            },
        }

    ep = progress_doc.get("episodes")
    ep_b = progress_doc.get("episode_budget")
    ticks = progress_doc.get("physical_ticks")
    tick_b = progress_doc.get("tick_budget")
    fraction = _progress_fraction(progress_doc, status)
    return {
        **{f"{name}_tasks": count for name, count in _bucket_counts([status]).items()},
        "deferred_tasks": 0,
        "total_tasks": 1,
        "fraction": round(fraction, 4) if fraction is not None else None,
        "episodes": {
            "min": int(ep) if isinstance(ep, (int, float)) else None,
            "max": int(ep) if isinstance(ep, (int, float)) else None,
            "budget": int(ep_b) if isinstance(ep_b, (int, float)) else None,
        },
        "physical_ticks": {
            "completed": int(ticks) if isinstance(ticks, (int, float)) else None,
            "budget": int(tick_b) if isinstance(tick_b, (int, float)) else None,
        },
    }


def summarize_run(root: Path, *, process_table=None) -> dict:
    """Summarize one run; `process_table` returns the host table for liveness checks."""
    root = Path(root).resolve()
    if process_table is None:
        from smartsom.experiments.control import processes as process_table
    plan = _read_json(root / "plan.json")
    run_doc = _read_json(root / "run.json")
    manifest = _read_json(root / "manifest.json")
    batch = _read_json(root / "batch.json")
    progress_doc = _read_json(root / "progress.json")
    runtime = _read_json(root / "control/runtime.json")
    live_snapshot = _read_json(root / "logs/progress.json")
    tracking = _read_json(root / "tracking/status.json")

    identity = (
        plan.get("run_id") or manifest.get("run_id") or run_doc.get("id") or str(root)
    )
    short_id = hashlib.sha256(str(identity).encode()).hexdigest()[:12]
    is_batch = plan.get("schema") == "smartsom.author-batch-plan/v1" or isinstance(
        plan.get("files"), list
    )
    if is_batch and runtime.get("inventory_extensions"):
        try:
            from smartsom.experiments.batch_inventory import extend_view

            plan, batch = extend_view(root, plan, batch, runtime)
        except (OSError, ValueError, KeyError):
            pass
    kind = (
        live_snapshot.get("kind")
        or run_doc.get("kind")
        or ("batch-directory" if is_batch else "single")
    )
    journal_status = runtime.get("status")
    ledger_status = (batch if is_batch else progress_doc).get("status")
    # The driver may fail before the prepared scheduler takes ownership.
    if (
        journal_status
        and ledger_status in RUNTIME_FINAL
        and journal_status != ledger_status
    ):
        # A finished prior journal can also predate a failed background relaunch.
        ledger_time = _best_mtime(
            root, ("batch.json" if is_batch else "progress.json",)
        )
        journal_time = _best_mtime(root, ("control/runtime.json",))
        if (journal_status == "prepared" or journal_status in RUNTIME_FINAL) and (
            ledger_time is not None
            and journal_time is not None
            and ledger_time >= journal_time
        ):
            journal_status = None
    status = (
        journal_status
        or ledger_status
        or batch.get("status")
        or live_snapshot.get("status")
        or run_doc.get("status")
        or "unknown"
    )
    dead_status = (
        dead_driver_status(
            root,
            process_table,
            _recorded_host(plan, run_doc, manifest),
            status,
            ledger_status,
        )
        if status in ACTIVE_STATUSES
        else None
    )
    alive_driver = dead_status is None
    if not alive_driver:
        status = dead_status
    stage = (
        live_snapshot.get("stage")
        or batch.get("stage")
        or progress_doc.get("stage")
        or run_doc.get("stage")
        or status
        if alive_driver
        else status
    )

    meta = (
        live_snapshot.get("meta")
        or plan.get("meta")
        or run_doc.get("meta")
        or manifest.get("meta")
    )
    if not isinstance(meta, dict):
        meta = resolved_meta(None, fallback_title=root.name, auto_summary=None)

    facts = (
        live_snapshot.get("facts")
        or plan.get("facts")
        or run_doc.get("facts")
        or manifest.get("facts")
        or {}
    )
    if (
        plan.get("incremental")
        and plan.get("facts")
        and isinstance(batch.get("selection"), dict)
    ):
        # The immutable plan describes launch. The verified extensions and latest
        # selection describe the inventory and requested seeds at this submission.
        rows = plan["files"]
        selected = set(batch["selection"]["tasks"])
        for receipt in (runtime.get("commands") or {}).values():
            if receipt.get("payload", {}).get("kind") == "add" and receipt.get(
                "revision", 0
            ) > batch.get("selection_revision", -1):
                selected.update(receipt.get("tasks", []))

        def seed(row):
            if "seed" in row:
                return row["seed"]
            # Older incremental descriptors store the seed in the canonical trial path.
            return int(Path(row["child"]).name.removeprefix("seed-"))

        selected_rows = [row for row in rows if row["id"] in selected]
        facts = {
            **plan["facts"],
            "task_count": len(rows),
            "selected_task_count": len(selected_rows),
            "seeds": {
                "all": list(dict.fromkeys(seed(row) for row in rows)),
                "selected": list(dict.fromkeys(seed(row) for row in selected_rows)),
            },
        }
        meta = resolved_meta(
            meta,
            fallback_title=(meta.get("effective") or {}).get("short_title")
            or root.name,
            auto_summary=build_auto_summary(facts),
        )
    updated_at = (
        runtime.get("updated_at")
        or live_snapshot.get("updated_at")
        or _best_mtime(
            root,
            (
                "logs/progress.json",
                "control/runtime.json",
                "progress.json",
                "batch.json",
                "run.json",
                "plan.json",
                "manifest.json",
            ),
        )
    )
    eta = live_snapshot.get("eta")
    if not isinstance(eta, dict):
        eta = placeholder_eta(updated_at=updated_at)

    progress = _aggregate_progress(
        root, plan, batch, runtime, progress_doc, live_snapshot, status, alive_driver
    )
    wandb_url = tracking.get("url")
    if not wandb_url and isinstance(live_snapshot.get("tasks"), list):
        for row in live_snapshot["tasks"]:
            url = (row.get("values") or {}).get("wandb_url")
            if url:
                wandb_url = url
                break

    return {
        "id": str(identity),
        "short_id": short_id,
        "directory": str(root),
        "kind": kind,
        "status": status,
        "stage": stage,
        "meta": meta,
        "facts": facts,
        "progress": progress,
        "eta": eta,
        "wandb_url": wandb_url,
        "updated_at": updated_at,
    }


def _category(status: str) -> int:
    if status in ACTIVE_STATUSES:
        return 0
    if status in QUEUED_STATUSES | PAUSED_STATUSES:
        return 1
    return 2


def collect_summary(
    roots=None, *, active_only: bool = False, recent: int | None = 20
) -> dict:
    """Build the unified `smartsom.monitor-summary/v1` payload."""
    cpu_primed = False
    t0 = time.monotonic()
    try:
        import psutil

        psutil.cpu_percent(interval=None)
        cpu_primed = True
    except Exception:
        pass

    from smartsom.experiments.control import processes

    process_table = cache(processes)  # One host scan, only if a run claims activity.
    records = []
    for directory in discover_top_level_runs(roots):
        try:
            records.append(summarize_run(directory, process_table=process_table))
        except (OSError, ValueError):
            continue

    buckets = _bucket_counts(buckets=map(_record_bucket, records))
    counts = {
        "running": buckets.pop("active"),
        **buckets,
        "total": len(records),
    }

    active_rows = sorted(
        [r for r in records if _category(r["status"]) < 2],
        key=lambda r: (_category(r["status"]), -(r.get("updated_at") or 0)),
    )
    if active_only:
        selected = active_rows
    else:
        terminal_rows = sorted(
            [r for r in records if _category(r["status"]) == 2],
            key=lambda r: -(r.get("updated_at") or 0),
        )
        if recent is not None:
            terminal_rows = terminal_rows[: max(0, recent)]
        selected = active_rows + terminal_rows

    if cpu_primed:
        elapsed = time.monotonic() - t0
        remaining = max(0.0, 0.1 - elapsed)
        if remaining > 0:
            time.sleep(remaining)
        host = host_resources(cpu_interval=None)
    else:
        host = host_resources()

    return {
        "schema": SUMMARY_SCHEMA,
        "generated_at": time.time(),
        "host": host,
        "counts": counts,
        "batches": selected,
    }


def render_summary_text(payload: dict) -> str:
    host = payload.get("host") or {}
    counts = payload.get("counts") or {}
    cpu_text = (
        f"{host['cpu_percent']}% ({host.get('cpus_used')}/{host.get('cpus_total')})"
        if host.get("cpu_percent") is not None
        else "N/A"
    )
    mem_text = (
        f"{host['memory_used_gib']}/{host['memory_total_gib']} GiB"
        if host.get("memory_used_gib") is not None
        else "N/A"
    )
    lines = [
        f"Host: {host.get('name')} ({host.get('platform')}) · CPU {cpu_text} · RAM {mem_text}",
        f"Runs: running={counts.get('running', 0)} queued={counts.get('queued', 0)} "
        f"paused={counts.get('paused', 0)} completed={counts.get('completed', 0)} "
        f"failed={counts.get('failed', 0)} stopped={counts.get('stopped', 0)} "
        f"total={counts.get('total', 0)}",
    ]
    for row in payload.get("batches", []):
        effective = (row.get("meta") or {}).get("effective") or {}
        title = effective.get("short_title") or Path(row["directory"]).name
        desc = effective.get("short_description") or ""
        prog = row.get("progress") or {}
        pct = (
            f"{prog['fraction'] * 100:.0f}%"
            if prog.get("fraction") is not None
            else "—"
        )
        ep = prog.get("episodes") or {}
        ep_text = ""
        if ep.get("budget"):
            if ep.get("min") is not None and ep.get("max") is not None:
                rng = (
                    f"{ep['min']}"
                    if ep["min"] == ep["max"]
                    else f"{ep['min']}-{ep['max']}"
                )
                ep_text = f" · ep {rng}/{ep['budget']}"
        lines.append(
            f"  [{row['status']}] {title}"
            + (f" — {desc}" if desc else "")
            + f" | {pct} ({prog.get('completed_tasks', 0)}/{prog.get('total_tasks', 0)} done"
            + f", {prog.get('active_tasks', 0)} active{ep_text})"
        )
    return "\n".join(lines)
