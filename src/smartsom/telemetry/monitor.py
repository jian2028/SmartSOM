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
                    return {
                        "H": "up",
                        "P": "down",
                        "K": "left",
                        "M": "right",
                        "I": "pageup",
                        "Q": "pagedown",
                    }.get(msvcrt.getwch())
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
    raw = os.read(fd, 1)
    if raw and raw[0] >= 0xC0:
        size = 2 if raw[0] < 0xE0 else 3 if raw[0] < 0xF0 else 4
        for _ in range(size - 1):
            if select.select([fd], [], [], 0.01)[0]:
                raw += os.read(fd, 1)
    key = raw.decode("utf-8", "replace")
    if key == "\x1b" and select.select([fd], [], [], 0.01)[0]:
        prefix = os.read(fd, 1)
        if prefix in (b"[", b"O"):
            final = None
            sequence = b""
            # CSI/SS3 arrows are not a literal '[' pagination key.
            for _ in range(12):
                if not select.select([fd], [], [], 0.01)[0]:
                    break
                final = os.read(fd, 1)
                sequence += final
                if final and 0x40 <= final[0] <= 0x7E:
                    break
            return {b"5~": "pageup", b"6~": "pagedown"}.get(sequence) or {
                b"A": "up",
                b"B": "down",
                b"C": "right",
                b"D": "left",
            }.get(final)
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


def _control_notice(display):
    from rich.text import Text

    display.publish(force=True)
    if not display.live:
        display.console.print(Text(display.notice), soft_wrap=True)


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
    limit_error = None
    search_buffer = None
    interaction_notice = None
    interaction_until = 0.0
    pending_command = None
    task_command_context = {}
    task_action_target = None
    queue_before = None
    last_global_key = float("-inf")
    remove_target = None
    remove_buffer = ""
    exit_menu = False
    exit_owner_id = None
    tracking_prompt = None
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
                    if controlling:
                        from smartsom.experiments.foreground import pending_prompt

                        tracking_prompt = pending_prompt(root)
                    if exit_menu:
                        prompt = (
                            "1 / Enter 停止并保留产物（默认）\n"
                            "2 停止后删除本地全部产物（包括模型与 tracking）\n"
                            "Esc / Ctrl+C 取消，继续运行\n"
                            "实验配置与 W&B 云端保留\n"
                            f"退出任务：{root}"
                        )
                    elif tracking_prompt:
                        prompt = (
                            f"W&B 启动失败：{tracking_prompt['error']}\n"
                            "y 明确选择离线继续 · n 拒绝离线 · q / Ctrl+C 退出菜单"
                        )
                    elif confirm_until > now:
                        prompt = "3 秒内再次 Ctrl+C 关闭监控"
                    elif remove_target is not None:
                        from smartsom.telemetry.tuning_dashboard import runtime_status

                        prompt = (
                            f"移除批次 {remove_target['batch_id']}\n"
                            f"任务 {remove_target['task_id']}\n"
                            f"状态 {runtime_status(remove_target.get('task', {}).get('status'))} · 安全保存后释放资源，保留文件\n"
                            f"输入 {remove_target['short_id']} 后 Enter 确认：{remove_buffer or '_'} · Esc 取消"
                        )
                    elif limit_buffer is not None:
                        prompt = (
                            limit_error + " · " if limit_error else ""
                        ) + f"总并发上限：{limit_buffer or '_'}（Enter 确认，Esc 取消）"
                    elif search_buffer is not None:
                        prompt = (
                            f"搜索任务：{search_buffer or '_'} · Enter 应用 · Esc 取消"
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
                    "exit"
                    if exit_menu
                    else "tracking"
                    if tracking_prompt
                    else "remove"
                    if remove_target is not None
                    else "limit"
                    if limit_buffer is not None
                    else "search"
                    if search_buffer is not None
                    else None
                )
                state = (display.stage, display.status, display.notice)
                display.publish(force=once or state != previous_state)
                if (
                    (exit_menu or tracking_prompt)
                    and state != previous_state
                    and not display.live
                ):
                    _control_notice(display)
                if task_action_target is None:
                    task_action_target = display.task_pages.anchor
                previous_state = state
                if once or (display.status in FINAL and not exit_menu):
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

                if exit_menu:
                    if interrupts[0] or key == "\x1b":
                        interrupts[0] = 0
                        exit_menu = False
                        continue
                    if key not in {"1", "2", "\r", "\n"}:
                        continue
                    from smartsom.experiments.control import stop

                    delete = key == "2"
                    display.notice = "正在请求安全停止，等待保存边界和所有所属进程退出；尚未删除任何文件"
                    _control_notice(display)
                    try:
                        result = stop(
                            root,
                            timeout=120,
                            delete=delete,
                            expected_owner_id=exit_owner_id,
                        )
                    except (OSError, ValueError) as exc:
                        display.notice = f"停止/删除未完成：{exc}"
                        _control_notice(display)
                        return 1
                    display.notice = (
                        "已停止并删除本地产物；实验配置与 W&B 云端记录保留"
                        if result.get("deleted")
                        else "等待停止超时：未删除文件，请用 status 检查后重试"
                        if result.get("remaining")
                        else "已停止，本地产物已保留"
                    )
                    _control_notice(display)
                    return int(bool(result.get("remaining")))

                if (
                    controlling
                    and (interrupts[0] or key == "q")
                    and search_buffer is None
                    and limit_buffer is None
                    and remove_target is None
                ):
                    from smartsom.experiments.control import read

                    owner = read(root)
                    exit_owner_id = owner["id"] if owner else None
                    interrupts[0] = 0
                    display.runtime_help = False
                    display.runtime_detail_task = None
                    display.runtime_queue = False
                    exit_menu = True
                    continue
                if tracking_prompt:
                    if key in {"y", "n"}:
                        from smartsom.experiments.foreground import answer_prompt

                        answer_prompt(root, tracking_prompt, offline=key == "y")
                    elif key == "d":
                        return 0
                    continue

                if search_buffer is not None:
                    if interrupts[0] or key == "\x1b":
                        interrupts[0] = 0
                        search_buffer = None
                    elif key in {"\r", "\n"}:
                        display.task_pages.query = search_buffer.strip()
                        display.task_pages.pending = 0
                        search_buffer = None
                        display.notice = None
                        display.publish(force=True)
                        task_action_target = display.task_pages.anchor
                        display.selection_notice = None
                    elif key in {"\x7f", "\b"}:
                        search_buffer = search_buffer[:-1]
                    elif (
                        key
                        and len(key) == 1
                        and key.isprintable()
                        and len(search_buffer) < 128
                    ):
                        search_buffer += key
                    continue

                if limit_buffer is not None and interrupts[0]:
                    interrupts[0] = 0
                    limit_buffer, limit_error = None, None
                    confirm_until = 0.0
                    continue

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
                    elif key in {"up", "down", "pageup", "pagedown"}:
                        amount = (
                            getattr(display, "runtime_confirmation_offset_page_size", 1)
                            if key in {"pageup", "pagedown"}
                            else 1
                        )
                        display.runtime_confirmation_offset = getattr(
                            display, "runtime_confirmation_offset", 0
                        ) + (amount if key in {"down", "pagedown"} else -amount)
                    elif key in {"\x7f", "\b"}:
                        remove_buffer = remove_buffer[:-1]
                    elif (
                        key
                        and len(key) == 1
                        and key.isprintable()
                        and len(remove_buffer) < 128
                    ):
                        remove_buffer += key
                    interaction_until = time.monotonic() + 5.0
                    continue
                if (
                    controlling
                    and runtime
                    and limit_buffer is None
                    and not getattr(display, "runtime_detail_task", None)
                    and not getattr(display, "runtime_help", False)
                    and (key == "Q" or getattr(display, "runtime_queue", False))
                ):
                    if getattr(display, "runtime_queue", False):
                        if key in {"Q", "\x1b"} or interrupts[0]:
                            interrupts[0] = 0
                            display.runtime_queue = False
                            if queue_before is not None:
                                (
                                    display.task_pages.scope,
                                    display.task_pages.query,
                                    display.task_pages.anchor,
                                ) = queue_before
                            queue_before = None
                            display.selection_notice = None
                        elif key in {"up", "down", "j", "k"}:
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
                                            index + (1 if key in {"down", "j"} else -1),
                                        ),
                                    )
                                ]
                            display.selection_notice = None
                            task_action_target = display.task_pages.anchor
                        elif key in {"\r", "\n", "left", "right"}:
                            from smartsom.experiments.runtime_control import submit

                            try:
                                target = task_action_target
                                if (
                                    not target
                                    or target != display.task_pages.anchor
                                    or target not in display.task_pages.ids
                                ):
                                    raise ValueError(
                                        "任务已离开排队视图，请重新选择后操作"
                                    )
                                action = (
                                    ("earlier" if key == "left" else "later")
                                    if key in {"left", "right"}
                                    else "remove"
                                    if target in runtime.get("priority_tasks", [])
                                    else "add"
                                )
                                receipt = submit(
                                    root,
                                    "priority",
                                    task_id=target,
                                    value=action,
                                    expected_revision=runtime.get("revision"),
                                )
                                pending_command = receipt.get(
                                    "id", receipt.get("command_id")
                                )
                                task_command_context[pending_command] = (
                                    f"优先队列 · {target} · {action}\n"
                                )
                                interaction_notice = "优先顺序请求已发送"
                            except (OSError, ValueError) as exc:
                                interaction_notice = str(exc)
                            interaction_until = time.monotonic() + 5
                        elif key == "?":
                            display.runtime_help = True
                            display.runtime_help_offset = 0
                        elif key == "d":
                            return 0
                    else:
                        queue_before = (
                            display.task_pages.scope,
                            display.task_pages.query,
                            display.task_pages.anchor,
                        )
                        display.runtime_queue = True
                        display.task_pages.scope, display.task_pages.query = (
                            "queued",
                            "",
                        )
                        display.task_pages.pending = 0
                        display.selection_notice = None
                    display.publish(force=True)
                    if key in {"Q", "\x1b"}:
                        task_action_target = display.task_pages.anchor
                    continue
                if (
                    controlling
                    and key == "D"
                    and limit_buffer is None
                    and not getattr(display, "runtime_detail_task", None)
                    and not getattr(display, "runtime_help", False)
                ):
                    from smartsom.experiments.runtime_control import removal_target

                    runtime = (display.tuning or {}).get("runtime_control", {})
                    target_id = task_action_target
                    try:
                        if target_id is None or target_id not in display.task_pages.ids:
                            raise ValueError(
                                "选中任务已不可用，请重新选择；未发送控制请求"
                            )
                        remove_target = removal_target(root, target_id)
                        remove_buffer = ""
                        display.runtime_confirmation_offset = 0
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
                    return 0
                if getattr(display, "runtime_help", False):
                    if key in {"\x1b", "?"}:
                        display.runtime_help = False
                    elif key == "d":
                        return 0
                    elif key in {"up", "down", "j", "k", "pageup", "pagedown"}:
                        amount = (
                            getattr(display, "runtime_help_offset_page_size", 1)
                            if key in {"pageup", "pagedown"}
                            else 1
                        )
                        display.runtime_help_offset = getattr(
                            display, "runtime_help_offset", 0
                        ) + (amount if key in {"down", "j", "pagedown"} else -amount)
                    display.publish(force=True)
                    continue
                if key == "?" and limit_buffer is None and runtime:
                    display.runtime_help = True
                    display.runtime_help_offset = 0
                    display.publish(force=True)
                    continue
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
                    elif key in {"pageup", "pagedown"}:
                        display.runtime_detail_offset += getattr(
                            display, "runtime_detail_offset_page_size", 1
                        ) * (1 if key == "pagedown" else -1)
                    elif key in {"g", "G"}:
                        display.runtime_detail_offset = 0 if key == "g" else 10**9
                    elif key == "r":
                        display.runtime_detail_raw = not getattr(
                            display, "runtime_detail_raw", False
                        )
                        display.runtime_detail_offset = 0
                    elif key == "d":
                        return 0
                    display.publish(force=True)
                    continue
                if limit_buffer is None and runtime and key == "/":
                    search_buffer = display.task_pages.query
                    continue
                if limit_buffer is None and runtime and key in {"e", "n"}:
                    previous = display.task_pages.anchor
                    display.task_pages.scope = (
                        "unfinished"
                        if key == "e" and display.task_pages.scope == "failed"
                        else "failed"
                    )
                    display.task_pages.pending = 0
                    if key == "n":
                        display.task_pages.query = ""
                    display.publish(force=True)
                    ids = display.task_pages.ids
                    if key == "n" and previous in ids:
                        display.task_pages.anchor = ids[
                            (ids.index(previous) + 1) % len(ids)
                        ]
                        display.publish(force=True)
                    task_action_target = display.task_pages.anchor
                    display.selection_notice = None
                    continue
                if (
                    key == "f"
                    and limit_buffer is None
                    and (display.tuning or {}).get("runtime_control")
                ):
                    display.task_pages.cycle_scope()
                    display.publish(force=True)
                    task_action_target = display.task_pages.anchor
                    display.selection_notice = None
                    continue
                if key in {"\r", "\n", "i"} and limit_buffer is None:
                    display.runtime_detail_task = display.task_pages.anchor
                    display.runtime_detail_offset = 0
                    display.runtime_detail_raw = False
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
                        display.selection_notice = None
                    continue
                if key in {"[", "]", "left", "right"} and limit_buffer is None:
                    if display.move_task_page(1 if key in {"]", "right"} else -1):
                        display.publish(force=True)
                        task_action_target = display.task_pages.anchor
                        display.selection_notice = None
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
                            limit_error = None
                        elif key in {"\x7f", "\b"}:
                            limit_buffer = limit_buffer[:-1]
                            limit_error = None
                        elif key in {"\r", "\n"}:
                            try:
                                if not limit_buffer or int(limit_buffer) < 1:
                                    limit_error = "请输入大于 0 的整数"
                                    continue
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
                                    raise ValueError(
                                        "historical queue controls require archived source"
                                    )
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
                        limit_error = None
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
