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
    from smartsom.experiments.runtime_control import snapshot as runtime_snapshot

    runtime = runtime_snapshot(root)
    if runtime:
        result.setdefault("tuning", {})["runtime_control"] = runtime
        result.update(status=runtime["status"])
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
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        except OSError:
            pass  # A disconnected terminal must not stop the scheduler.


def _key(fd, timeout):
    if fd == "windows-console":
        import msvcrt

        deadline = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in {"\x00", "\xe0"}:
                    return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(
                        msvcrt.getwch()
                    )
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
            final = None
            # CSI/SS3 arrows are not a literal '[' pagination key.
            for _ in range(12):
                if not select.select([fd], [], [], 0.01)[0]:
                    break
                final = os.read(fd, 1)
                if final and 0x40 <= final[0] <= 0x7E:
                    break
            return {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}.get(final)
    return key


def _receipt_notice(receipt):
    identity = receipt.get("id", receipt.get("command_id"))
    detail = receipt.get("reason") or receipt.get("result")
    return f"命令 {identity}: {receipt.get('status', 'received')}" + (
        " · " + str(detail) if detail else ""
    )


def _remember_control(display, receipt, kind, task_id=None):
    if kind not in {"pause", "resume", "remove"}:
        return
    row = dict(receipt)
    row.setdefault("payload", {"kind": kind, "task_id": task_id})
    identity = row.get("id", row.get("command_id"))
    local = dict(getattr(display, "runtime_local_commands", {}))
    local[identity] = row
    display.runtime_local_commands = local
    display.runtime_feedback_id = identity


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
    pending_command = None
    task_command_context = {}
    task_action_target = None
    last_global_key = float("-inf")
    remove_target = None
    remove_buffer = ""
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
                    runtime = (display.tuning or {}).get("runtime_control", {})
                    if pending_command:
                        commands = runtime.get("commands", {})
                        receipt = (
                            commands.get(pending_command)
                            if isinstance(commands, dict)
                            else next(
                                (
                                    item
                                    for item in commands
                                    if item.get("id") == pending_command
                                ),
                                None,
                            )
                        )
                        if receipt and (receipt.get("payload") or {}).get("kind") in {
                            "pause",
                            "resume",
                            "remove",
                        }:
                            pending_command = None
                        elif receipt:
                            interaction_notice = task_command_context.get(
                                pending_command, ""
                            ) + _receipt_notice(receipt)
                            interaction_until = now + 5.0
                            if receipt.get("status") in {"applied", "rejected"}:
                                pending_command = None
                    if now >= interaction_until:
                        interaction_notice = None
                    age = max(0, time.time() - display.updated_at)
                    stale = (
                        age > 5
                        and display.status not in FINAL
                        and display.status != "stopping"
                    )
                    display.observation_notice = (
                        f"观测过期 {age:.0f}s · 进程状态未知"
                        if stale and runtime
                        else None
                    )
                    if stale and not runtime:
                        display.notice = f"Last recorded update {age:.0f}s ago; process state unknown"
                    prompt = None
                    if confirm_until > now:
                        prompt = (
                            "3 秒内再次 Ctrl+C 安全停止任务"
                            if controlling
                            else "3 秒内再次 Ctrl+C 关闭监控"
                        )
                    elif remove_target is not None:
                        prompt = (
                            f"移除批次 {remove_target['batch_id']}\n"
                            f"任务 {remove_target['task_id']}\n"
                            f"状态 {remove_target.get('task', {}).get('status', 'unknown')} · 安全保存后释放资源，保留文件\n"
                            f"输入 {remove_target['short_id']} 后 Enter 确认：{remove_buffer or '_'} · Esc 取消"
                        )
                    elif limit_buffer is not None:
                        prompt = (
                            f"总并发上限：{limit_buffer or '_'}（Enter 确认，Esc 取消）"
                        )
                    elif interaction_notice:
                        prompt = interaction_notice
                    if prompt:
                        display.notice = (
                            display.notice + "\n" + prompt
                            if stale and not runtime
                            else prompt
                        )
                except (OSError, ValueError, KeyError, TypeError):
                    if (display.tuning or {}).get("runtime_control"):
                        display.observation_notice = "快照暂不可用 · 进程状态未知"
                    else:
                        display.notice = "Snapshot temporarily unavailable; showing last valid update"
                display.control_input = (
                    "remove"
                    if remove_target is not None
                    else "limit"
                    if limit_buffer is not None
                    else None
                )
                state = (display.stage, display.status, display.notice)
                display.publish(force=once or state != previous_state)
                if task_action_target is None:
                    task_action_target = display.task_pages.anchor
                previous_state = state
                if once or display.status in FINAL:
                    return 0
                try:
                    key = _key(fd, poll_seconds)
                except KeyboardInterrupt:
                    interrupts[0] += 1
                    key = None
                except (OSError, EOFError):
                    return 0
                if key == "":
                    return 0

                if remove_target is not None:
                    if interrupts[0] or key == "\x1b":
                        interrupts[0] = 0
                        remove_target, remove_buffer = None, ""
                        interaction_notice = "移除已取消"
                    elif key in {"\r", "\n"}:
                        if remove_buffer == remove_target["short_id"]:
                            from smartsom.experiments.runtime_control import submit

                            try:
                                receipt = submit(
                                    root,
                                    "remove",
                                    task_id=remove_target["task_id"],
                                    expected_revision=remove_target["revision"],
                                    confirmation=remove_target["confirmation"],
                                )
                                pending_command = receipt.get(
                                    "id", receipt.get("command_id")
                                )
                                _remember_control(
                                    display, receipt, "remove", remove_target["task_id"]
                                )
                                interaction_notice = None
                            except (ValueError, OSError) as exc:
                                interaction_notice = str(exc)
                        else:
                            interaction_notice = "目标未匹配，移除已取消"
                        remove_target, remove_buffer = None, ""
                    elif key in {"\x7f", "\b"}:
                        remove_buffer = remove_buffer[:-1]
                    elif key and key.isprintable() and len(remove_buffer) < 128:
                        remove_buffer += key
                    interaction_until = time.monotonic() + 5.0
                    continue
                if (
                    controlling
                    and key == "D"
                    and limit_buffer is None
                    and not getattr(display, "runtime_detail_task", None)
                ):
                    from smartsom.experiments.runtime_control import removal_target

                    runtime = (display.tuning or {}).get("runtime_control", {})
                    target_id = display.task_pages.anchor
                    if target_id is None:
                        target_id = next(iter(runtime.get("tasks", {})), None)
                    try:
                        if target_id is None:
                            raise ValueError("没有可选任务")
                        remove_target = removal_target(root, target_id)
                        remove_buffer = ""
                    except (ValueError, OSError) as exc:
                        interaction_notice = str(exc)
                        interaction_until = time.monotonic() + 5.0
                    continue
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
                        if (display.tuning or {}).get("runtime_control"):
                            from smartsom.experiments.runtime_control import submit

                            submit(root, "stop")
                        else:
                            stop(root, timeout=0)
                        display.notice = "已请求安全停止，等待保存边界"
                        interaction_notice = display.notice
                    except ValueError as exc:
                        display.notice = str(exc)
                        interaction_notice = display.notice
                    interaction_until = time.monotonic() + 5.0
                    display.publish(force=True)
                if controlling and limit_buffer is None and key in {"P", "R"}:
                    runtime = (display.tuning or {}).get("runtime_control", {})
                    if runtime:
                        from smartsom.experiments.runtime_control import submit

                        target = (
                            getattr(display, "runtime_detail_task", None)
                            or task_action_target
                        )
                        action = "pause" if key == "P" else "resume"
                        context = f"任务 {target} · {action} 请求（状态以观测为准）\n"
                        try:
                            if not target:
                                raise ValueError("未选中任务，未发送控制请求")
                            if (
                                target not in display.task_pages.ids
                                or runtime.get("tasks", {})
                                .get(target, {})
                                .get("status")
                                == "removed"
                            ):
                                raise ValueError(
                                    "选中任务已不可用，请重新选择；未发送控制请求"
                                )
                            from smartsom.telemetry.tuning_dashboard import (
                                pending_pause,
                            )

                            receipt = (
                                pending_pause(display, target)
                                if action == "pause"
                                else None
                            ) or submit(root, action, task_id=target)
                            _remember_control(display, receipt, action, target)
                            pending_command = receipt.get(
                                "id", receipt.get("command_id")
                            )
                            task_command_context[pending_command] = context
                            interaction_notice = None
                        except (OSError, ValueError) as exc:
                            interaction_notice = context + str(exc)
                        interaction_until = time.monotonic() + 5.0
                        display.notice = interaction_notice
                        display.publish(force=True)
                        continue
                if key == "a" and limit_buffer is None:
                    from smartsom.telemetry.tuning_dashboard import (
                        runtime_feedback,
                        runtime_feedback_commands,
                    )

                    commands = runtime_feedback_commands(display)
                    if commands:
                        current = runtime_feedback(display)
                        index = commands.index(current) if current in commands else -1
                        chosen = commands[(index + 1) % len(commands)]
                        display.runtime_feedback_id = chosen.get(
                            "id", chosen.get("command_id")
                        )
                        display.publish(force=True)
                    continue
                if limit_buffer is None and getattr(
                    display, "runtime_detail_task", None
                ):
                    if key == "\x1b":
                        display.runtime_detail_task = None
                    elif key in {"up", "down", "j", "k"}:
                        display.runtime_detail_offset += (
                            1 if key in {"down", "j"} else -1
                        )
                    elif key == "d":
                        return 0
                    display.publish(force=True)
                    continue
                if key in {"\r", "\n", "i"} and limit_buffer is None:
                    display.runtime_detail_task = display.task_pages.anchor
                    display.runtime_detail_offset = 0
                    display.publish(force=True)
                    continue
                if key in {"j", "k", "up", "down"} and limit_buffer is None:
                    ids = display.task_pages.ids
                    if ids:
                        index = (
                            ids.index(display.task_pages.anchor)
                            if display.task_pages.anchor in ids
                            else 0
                        )
                        display.task_pages.anchor = ids[
                            max(
                                0,
                                min(
                                    len(ids) - 1,
                                    index + (1 if key in {"j", "down"} else -1),
                                ),
                            )
                        ]
                        display.publish(force=True)
                        task_action_target = display.task_pages.anchor
                    continue
                if key in {"[", "]", "left", "right"} and limit_buffer is None:
                    if display.move_task_page(1 if key in {"]", "right"} else -1):
                        display.publish(force=True)
                        task_action_target = display.task_pages.anchor
                    continue
                runtime = (display.tuning or {}).get("runtime_control", {})
                if controlling and (runtime or (display.tuning or {}).get("manual")):
                    if limit_buffer is None and runtime and key in {"+", "-", " "}:
                        from smartsom.experiments.runtime_control import submit

                        try:
                            if key == " ":
                                from smartsom.telemetry.tuning_dashboard import (
                                    pending_pause,
                                )

                                now = time.monotonic()
                                gap = now - last_global_key
                                last_global_key = now
                                # A held/buffered Space must not reverse a pause once
                                # its observation happens to finish during the burst.
                                if gap < 0.5:
                                    continue
                                existing = pending_pause(display)
                                action = (
                                    "pause"
                                    if existing or not runtime.get("paused")
                                    else "resume"
                                )
                                receipt = existing or submit(root, action)
                                _remember_control(display, receipt, action)
                                interaction_notice = None
                            else:
                                receipt = submit(
                                    root, "delta", value=1 if key == "+" else -1
                                )
                                interaction_notice = _receipt_notice(receipt)
                            pending_command = receipt.get(
                                "id", receipt.get("command_id")
                            )
                        except (OSError, ValueError) as exc:
                            interaction_notice = str(exc)
                        interaction_until = time.monotonic() + 5.0
                        display.notice = interaction_notice
                        display.publish(force=True)
                        continue
                    if limit_buffer is not None:
                        if key and key.isdecimal() and len(limit_buffer) < 4:
                            limit_buffer += key
                        elif key in {"\x7f", "\b"}:
                            limit_buffer = limit_buffer[:-1]
                        elif key in {"\r", "\n"}:
                            from smartsom.experiments.live_queue import set_limit

                            try:
                                if runtime:
                                    from smartsom.experiments.runtime_control import (
                                        submit,
                                    )

                                    receipt = submit(
                                        root, "limit", value=int(limit_buffer)
                                    )
                                    pending_command = receipt.get(
                                        "id", receipt.get("command_id")
                                    )
                                    display.notice = _receipt_notice(receipt)
                                else:
                                    set_limit(root, int(limit_buffer))
                                    display.notice = f"已请求总并发上限 {limit_buffer}"
                                interaction_notice = display.notice
                            except (OSError, ValueError) as exc:
                                display.notice = str(exc)
                                interaction_notice = display.notice
                            interaction_until = time.monotonic() + 5.0
                            limit_buffer = None
                        elif key == "\x1b":
                            limit_buffer = None
                        display.control_input = (
                            "limit" if limit_buffer is not None else None
                        )
                        if limit_buffer is not None:
                            display.notice = f"总并发上限：{limit_buffer or '_'}（Enter 确认，Esc 取消）"
                        display.publish(force=True)
                        continue
                    if key == "c":
                        interaction_notice = None
                        limit_buffer = ""
                        display.control_input = "limit"
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
