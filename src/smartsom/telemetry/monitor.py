"""Read-only monitor for mainline runtime snapshots and historical manifests."""

import json
import math
import os
import select
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from smartsom._filesystem import native_path
from smartsom.telemetry.runtime import FINAL, SCHEMA, DisplayOptions, RuntimeDisplay

MAINLINE = {
    "smartsom.author-run/v1",
    "smartsom.author-batch-run/v1",
    "smartsom.live-queue-run/v1",
    "smartsom.tune-run/v1",
    "smartsom.experiment/v2",
    "smartsom.evaluation/v1",
    "smartsom.production-run/v1",
    "smartsom.study-manifest/v1",
}


def _count(value):
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _read_snapshot(root):
    root = Path(root)
    path = root / "logs/progress.json"
    if native_path(path).is_file():
        result = json.loads(native_path(path).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("invalid runtime progress snapshot")
        if result.get("schema") != SCHEMA:
            raise ValueError("unsupported runtime progress schema")
        if (
            not isinstance(result.get("tasks"), list)
            or result.get("updated_at") is None
            or not _count(result.get("updated_at"))
            or not _count(result.get("total_tasks"))
        ):
            raise ValueError("invalid runtime progress snapshot")
        if not all(
            isinstance(result.get(key), str)
            for key in ("name", "kind", "stage", "status")
        ) or any(
            not isinstance(row, dict)
            or not all(
                isinstance(row.get(key), str)
                for key in ("id", "name", "stage", "status")
            )
            or not isinstance(row.get("values"), dict)
            or not isinstance(row.get("learner", {}), dict)
            or not _count(row.get("total"))
            or not _count(row.get("completed"))
            or not _count(row.get("stage_started_at"))
            or not _count(row.get("started_at"))
            for row in result["tasks"]
        ):
            raise ValueError("invalid runtime progress snapshot")
        from smartsom.telemetry.execution_metrics import (
            validate_estimate,
            validate_observation,
        )

        for row in result["tasks"]:
            if row.get("execution") is not None:
                row["execution"] = validate_observation(row["execution"])
        if result.get("execution") is not None:
            if not isinstance(result["execution"], dict):
                raise ValueError("invalid execution presentation")
            if result["execution"].get("batch") is not None:
                validate_estimate(result["execution"]["batch"])
            for estimate in result["execution"].get("tasks", {}).values():
                validate_estimate(estimate)
            for key in ("experiment", "aggregate"):
                if result["execution"].get(key) is not None:
                    result["execution"][key] = validate_observation(
                        result["execution"][key]
                    )
        overview = result.get("overview")
        if overview is not None and (
            not isinstance(overview, dict)
            or any(
                not _count(overview.get(key))
                for key in (
                    "work_completed",
                    "work_total",
                    "training_completed",
                    "training_total",
                    "elapsed_seconds",
                    "eta_seconds",
                )
            )
        ):
            raise ValueError("invalid runtime overview")
        from smartsom.telemetry.workflow import validate_workflow

        if result.get("workflow") is not None:
            validate_workflow(result["workflow"])
        for row in result["tasks"]:
            if row["values"].get("workflow") is not None:
                validate_workflow(row["values"]["workflow"])
        if result.get("tuning") is not None:
            from smartsom.telemetry.tuning_dashboard import clean_summary

            result["tuning"] = clean_summary(result["tuning"])
        return result
    manifest = root / "run.json"
    if not manifest.exists():
        manifest = root / "manifest.json"
    if not manifest.exists():
        raise ValueError(
            "no mainline run manifest or runtime progress snapshot; experimental formats are unsupported"
        )
    record = json.loads(native_path(manifest).read_text(encoding="utf-8"))
    if record.get("schema") not in MAINLINE:
        raise ValueError(
            "unsupported run format; W38, dispatch pilot and overnight are not supported"
        )
    status = record.get("status", "unknown")
    tasks = []
    events = root / "logs/events.jsonl"
    if events.is_file():
        # Read a bounded tail, tolerate a partially written last line.
        with events.open("rb") as stream:
            stream.seek(max(0, events.stat().st_size - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            from smartsom.telemetry.runtime import LABELS

            tasks.append(
                {
                    "id": str(root),
                    "name": record.get("name", root.name),
                    "status": status,
                    "stage": event.get("stage", status),
                    "values": {k: v for k, v in event.items() if k in LABELS},
                }
            )
            break
    return {
        "schema": SCHEMA,
        "name": record.get("name", root.name),
        "kind": record.get("kind", "historical"),
        "status": status,
        "stage": record.get("stage", status),
        "updated_at": manifest.stat().st_mtime,
        "tasks": tasks,
        "total_tasks": None,
        "notice": "Historical data: only recorded fields are available",
    }


def read_snapshot(root):
    result = _read_snapshot(root)
    from smartsom.experiments.control import ACTIVE, alive, processes, read, requested

    owner = read(root)
    if owner:
        state = owner["status"]
        if state in ACTIVE and (requested(root) or state == "stopping"):
            identities = [owner["owner"], *owner["members"]]
            table = processes()
            if any(alive(item, table) for item in identities):
                state = "stopping"
            else:
                # The durable ledger can finish before the last Rich frame.
                ledger = Path(root) / "batch.json"
                if not native_path(ledger).is_file():
                    ledger = Path(root) / "run.json"
                try:
                    persisted = json.loads(
                        native_path(ledger).read_text(encoding="utf-8")
                    ).get("status")
                except (OSError, ValueError):
                    persisted = None
                state = (
                    persisted
                    if persisted in FINAL
                    else result["status"]
                    if result["status"] in FINAL
                    else "stopped"
                )
        if state in FINAL | {"stopping"}:
            result.update(status=state, stage=state)
            control_notice = (
                "Control: " + state + "; recovery uses the latest committed checkpoint"
            )
            result["notice"] = (
                result["notice"] + " · " + control_notice
                if result.get("notice")
                else control_notice
            )
    return result


@contextmanager
def _terminal_keys():
    if not sys.stdin.isatty():
        yield None
        return
    if os.name == "nt":
        yield "windows-console"
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _key(fd, timeout):
    if fd == "windows-console":
        import msvcrt

        deadline = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in {"\x00", "\xe0"}:
                    msvcrt.getwch()
                    return None
                if key == "\x03":
                    raise KeyboardInterrupt
                return key
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.02, remaining))
    if fd is None:
        time.sleep(timeout)
        return None
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    key = os.read(fd, 1).decode("utf-8", "ignore")
    if key == "\x1b" and select.select([fd], [], [], 0.01)[0]:
        prefix = os.read(fd, 1)
        if prefix in (b"[", b"O"):
            # CSI/SS3 arrows are not a literal '[' pagination key.
            for _ in range(12):
                if not select.select([fd], [], [], 0.01)[0]:
                    break
                final = os.read(fd, 1)
                if final and 0x40 <= final[0] <= 0x7E:
                    break
            return None
    return key


def monitor(root, *, once=False, options=None, poll_seconds=1.0, controlling=False):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("monitor requires an existing run directory")
    first = read_snapshot(root)
    display = RuntimeDisplay(
        options or DisplayOptions(), kind=first["kind"], readonly=True
    )
    display.controlling = controlling
    if not once and sys.stdin.isatty():
        display.start()
    previous_state = None
    confirm_until = 0.0
    requested_coverage = None
    limit_buffer = None
    interaction_notice = None
    interaction_until = 0.0
    interrupts = [0]
    prior_signal = None
    if hasattr(signal, "SIGINT"):
        try:
            prior_signal = signal.signal(
                signal.SIGINT, lambda *_: interrupts.__setitem__(0, interrupts[0] + 1)
            )
        except ValueError:  # a caller may run the read-only monitor in a thread
            pass
    try:
        with _terminal_keys() as fd:
            while True:
                try:
                    snapshot = first if first is not None else read_snapshot(root)
                    first = None
                    display.from_snapshot(snapshot)
                    now = time.monotonic()
                    if now >= interaction_until:
                        interaction_notice = None
                    age = max(0, time.time() - display.updated_at)
                    stale = (
                        age > 5
                        and display.status not in FINAL
                        and display.status != "stopping"
                    )
                    if stale:
                        display.notice = f"Last recorded update {age:.0f}s ago; process state unknown"
                    prompt = None
                    if confirm_until > now:
                        prompt = (
                            "3 秒内再次 Ctrl+C 安全停止任务"
                            if controlling
                            else "3 秒内再次 Ctrl+C 关闭监控"
                        )
                    elif limit_buffer is not None:
                        prompt = (
                            f"总并发上限：{limit_buffer or '_'}（Enter 确认，Esc 取消）"
                        )
                    elif interaction_notice:
                        prompt = interaction_notice
                    if prompt:
                        display.notice = (
                            display.notice + "\n" + prompt if stale else prompt
                        )
                except (OSError, ValueError, KeyError, TypeError):
                    display.notice = (
                        "Snapshot temporarily unavailable; showing last valid update"
                    )
                state = (display.stage, display.status, display.notice)
                display.publish(force=once or state != previous_state)
                previous_state = state
                if once or display.status in FINAL:
                    return 0
                try:
                    key = _key(fd, poll_seconds)
                except KeyboardInterrupt:
                    interrupts[0] += 1
                    key = None
                while interrupts[0]:
                    interrupts[0] -= 1
                    if time.monotonic() >= confirm_until:
                        confirm_until = time.monotonic() + 3.0
                        interaction_notice = None
                        display.notice = "再次 Ctrl+C 确认"
                        display.publish(force=True)
                        continue
                    confirm_until = 0.0
                    if not controlling:
                        return 0
                    from smartsom.experiments.control import stop

                    try:
                        stop(root, timeout=0)
                        display.notice = "已请求安全停止，等待保存边界"
                        interaction_notice = display.notice
                    except ValueError as exc:
                        display.notice = str(exc)
                        interaction_notice = display.notice
                    interaction_until = time.monotonic() + 5.0
                    display.publish(force=True)
                if key in {"[", "]"} and limit_buffer is None:
                    if display.move_task_page(1 if key == "]" else -1):
                        display.publish(force=True)
                    continue
                if controlling and (display.tuning or {}).get("manual"):
                    if limit_buffer is not None:
                        if key and key.isdecimal() and len(limit_buffer) < 4:
                            limit_buffer += key
                        elif key in {"\x7f", "\b"}:
                            limit_buffer = limit_buffer[:-1]
                        elif key in {"\r", "\n"}:
                            from smartsom.experiments.live_queue import set_limit

                            try:
                                set_limit(root, int(limit_buffer))
                                display.notice = f"已请求总并发上限 {limit_buffer}"
                                interaction_notice = display.notice
                            except ValueError as exc:
                                display.notice = str(exc)
                                interaction_notice = display.notice
                            interaction_until = time.monotonic() + 5.0
                            limit_buffer = None
                        elif key == "\x1b":
                            limit_buffer = None
                        if limit_buffer is not None:
                            display.notice = f"总并发上限：{limit_buffer or '_'}（Enter 确认，Esc 取消）"
                        display.publish(force=True)
                        continue
                    if key == "c":
                        interaction_notice = None
                        limit_buffer = ""
                        display.notice = "输入总并发上限，Enter 确认，Esc 取消"
                        display.publish(force=True)
                        continue
                if key == "d":
                    return 0
                if key == "p" and controlling:
                    from smartsom.experiments.control import set_preflight_coverage

                    preflight = display.preflight or {}
                    coverage = requested_coverage or preflight.get("coverage")
                    target = "representative" if coverage == "each" else "skip"
                    try:
                        set_preflight_coverage(root, target)
                        requested_coverage = target
                        display.notice = f"已请求将烟测范围改为 {target}"
                    except ValueError as exc:
                        display.notice = str(exc)
                    display.publish(force=True)
    finally:
        if prior_signal is not None:
            signal.signal(signal.SIGINT, prior_signal)
        display.close()
