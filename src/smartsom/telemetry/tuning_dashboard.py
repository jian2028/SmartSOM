"""Read-only presentation of calibration and trial resource facts; no Ray imports."""

import json
import math
import time
from copy import deepcopy

from rich import box
from rich.console import Group
from rich.panel import Panel

from smartsom.telemetry.dashboard import (
    PHASE_NAMES,
    bar,
    card,
    line,
    phase_bar,
    task_eta,
)
from smartsom.telemetry.runtime import FINAL, shown
from smartsom.telemetry.study_progress import duration
from smartsom.telemetry.task_pages import (
    TaskIdentity,
    render_panel,
    render_runtime_panel,
    runtime_progress,
)
from smartsom.telemetry.timeline import detail as timeline_detail
from smartsom.telemetry.timeline import has_calibration_stage, online_mode
from smartsom.telemetry.timeline import render as render_timeline

STAGES = {
    "preflight": "预检",
    "smoke": "逐条烟测",
    "calibration": "性能评估",
    "calibrating": "性能评估",
    "waiting_resources": "等待资源",
    "waiting_for_resources": "等待资源",
    "training": "正式实验",
    "running": "正式实验",
    "completed": "已结束",
    "failed": "失败",
    "interrupted": "已中断",
    "partial_stopped": "部分任务已停止",
    "paused": "已暂停",
    "pausing": "等待安全暂停边界",
    "draining": "自然收缩中",
    "stopping": "正在停止",
}
PROCESS_FIELDS = {"pid", "name", "cpu_cores", "rss", "protected"}
COUNT_FIELDS = {
    "active_seconds",
    "wall_seconds",
    "remaining_seconds",
    "limit_seconds",
    "waiting_seconds",
    "cpus_available",
    "cpus_total",
    "cpus_effective",
    "host_cpu_total",
    "cpus_used",
    "owned_cpu_cores",
    "owned_memory_bytes",
    "cpu_age_seconds",
    "cpu_observed_at",
    "memory_available",
    "external_cpu_load",
    "threads",
    "requested_cpus",
    "actual_cpus",
    "allocation_epoch",
    "pid",
    "cpu_cores",
    "rss",
    "active",
    "limit",
    "ceiling",
    "cached_hint",
    "best_concurrency",
    "best_throughput",
    "latest_throughput",
    "measured_concurrency",
    "window_count",
}
WAITING = {"waiting_resources", "waiting_for_resources"}


def _json(value):
    """Reject nonfinite numbers and terminal controls, including nested DTOs."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("invalid tuning number")
    elif isinstance(value, str):
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("invalid tuning text")
    elif isinstance(value, list):
        for item in value:
            _json(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("invalid tuning field")
            _json(key)
            _json(item)
    else:
        raise ValueError("invalid tuning value")


def _counts(section):
    for key in COUNT_FIELDS:
        value = section.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid tuning count: " + key)


def clean_summary(summary):
    """Detach a JSON DTO and keep process identity limited to name and PID."""
    if not isinstance(summary, dict):
        raise ValueError("invalid tuning summary")
    result = deepcopy(summary)
    for name in ("calibration", "resources"):
        section = result.get(name, {})
        if not isinstance(section, dict):
            raise ValueError("invalid tuning " + name)
        _counts(section)
    resources = result.get("resources", {})
    processes = resources.get("processes", [])
    if not isinstance(processes, list):
        raise ValueError("invalid tuning processes")
    detached = []
    for process in processes:
        if not isinstance(process, dict):
            raise ValueError("invalid tuning process")
        process = {
            key: value for key, value in process.items() if key in PROCESS_FIELDS
        }
        _counts(process)
        if process.get("name") is not None and not isinstance(process["name"], str):
            raise ValueError("invalid tuning process name")
        if process.get("protected") is not None and not isinstance(
            process["protected"], bool
        ):
            raise ValueError("invalid tuning protected flag")
        detached.append(process)
    if "resources" in result:
        resources["processes"] = detached
    entries = result.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("invalid tuning entries")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("experiment_id"), str
        ):
            raise ValueError("invalid tuning entry")
        _counts(entry)
        for key in ("status", "resource_change_reason"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                raise ValueError("invalid tuning entry text")
        if entry.get("pending_resize") is not None and not isinstance(
            entry["pending_resize"], (bool, dict)
        ):
            raise ValueError("invalid tuning resize request")
    if "stage" in result and not isinstance(result["stage"], str):
        raise ValueError("invalid tuning stage")
    online = result.get("online", {})
    if not isinstance(online, dict):
        raise ValueError("invalid tuning online groups")
    for group in online.values():
        if not isinstance(group, dict):
            raise ValueError("invalid tuning online group")
        _counts(group)
    for key in ("candidates", "measured"):
        value = result.get("calibration", {}).get(key)
        if value is not None and not isinstance(value, (list, int)):
            raise ValueError("invalid tuning candidate count")
        if isinstance(value, bool) or isinstance(value, int) and value < 0:
            raise ValueError("invalid tuning candidate count")
    _json(result)
    return result


def candidate_counts(calibration):
    def count(value):
        return len(value) if isinstance(value, list) else value

    return count(calibration.get("measured")), count(calibration.get("candidates"))


def compact(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list))
        else shown(value)
    )


def memory(value):
    return "N/A" if value is None else f"{value / 1024**3:.2f} GiB"


def seconds_text(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return shown(value)


def metric_text(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return shown(value)


def suggestions(summary):
    """Offer manual app management only while blocked, never a process action."""
    if summary.get("stage") not in WAITING:
        return []
    resources = summary.get("resources", {})
    if not resources.get("external_cpu_load"):
        return []
    result = []
    for process in resources.get("processes", []):
        name = process.get("name") or ""
        protected_name = name.casefold()
        if (
            process.get("protected") is not False
            or not name
            or not process.get("cpu_cores")
            or any(
                marker in protected_name
                for marker in (
                    "smartsom",
                    "system",
                    "kernel",
                    "windowserver",
                    "launchd",
                    "loginwindow",
                    "kworker",
                )
            )
        ):
            continue
        result.append(process)
    return sorted(result, key=lambda item: item.get("cpu_cores", 0), reverse=True)[:3]


def resource_text(resources):
    return (
        f"可用 CPU {metric_text(resources.get('cpus_available'))} cores · "
        f"内存 {memory(resources.get('memory_available'))} · "
        f"外部 CPU 负载 {metric_text(resources.get('external_cpu_load'))} · "
        f"模式 {shown(resources.get('mode'))}"
    )


def allocation_text(entry):
    if entry.get("status") in FINAL:
        return "正式实验资源已释放"
    if entry.get("threads") is None and entry.get("actual_cpus") is None:
        return (
            "进程运行中；分配信息未记录"
            if entry.get("status") == "running"
            else "尚未分配正式实验资源"
        )
    value = (
        f"线程 {shown(entry.get('threads'))} · "
        f"CPU 请求 {shown(entry.get('requested_cpus'))} / 实际 {shown(entry.get('actual_cpus'))} · "
        f"分配版本 {shown(entry.get('allocation_epoch'))}"
    )
    if entry.get("pending_resize"):
        value += " · 已请求，待生效"
    return value


def actual_mode(row):
    """Read the worker-reported execution layout, not its calibration proposal."""
    values = row.get("values", {})
    raw = values.get("runtime_mode") or (values.get("workflow") or {}).get(
        "runtime_mode", ""
    )
    if not isinstance(raw, str):
        return {}
    result = {}
    for part in raw.replace(" · ", ";").split(";"):
        key, separator, value = part.strip().partition("=")
        if separator:
            result[key.strip()] = value.strip()
    return result


def actual_allocation(pairs, *, profile_mode=None):
    running = [
        (entry, row)
        for entry, row in pairs
        if entry.get("status") == "running" and row.get("status") == "running"
    ]
    reported = running or [(entry, row) for entry, row in pairs if actual_mode(row)]
    modes = [actual_mode(row) for _, row in reported]
    if not modes or not all(modes):
        return "实际资源：等待运行实例上报"
    fields = (
        "device",
        "num_envs",
        "envs",
        "sampling_processes",
        "threads",
        "numerical_threads",
    )
    if any(
        any(mode.get(key) != modes[0].get(key) for key in fields) for mode in modes[1:]
    ):
        return "实际资源：各实验配置不同（详见实例记录）"
    runtime = modes[0]
    device = {"cuda": "GPU/CUDA", "mps": "GPU/MPS"}.get(
        runtime.get("device"), runtime.get("device", "未知设备").upper()
    )
    envs = runtime.get("num_envs", runtime.get("envs", "?"))
    samplers = runtime.get("sampling_processes", "?")
    threads = runtime.get("numerical_threads", runtime.get("threads", "?"))
    concurrency = f" · 并行实验 {len(running)}" if running else ""
    allocation = (
        f" · {profile_mode.capitalize()}"
        if profile_mode in {"balanced", "performance"}
        else ""
    )
    return (
        f"{'实际运行' if running else '本次执行'} {device}{allocation}{concurrency} · 每实验环境 {envs}"
        f" · 采样进程 {samplers} · 计算线程 {threads}"
    )


def execution_lines(summary):
    resources = summary.get("resources", {})

    def measured(value, status=None, reason=None):
        if status in {"unavailable", "stale"}:
            return "观测不可用" if status == "unavailable" else "已过期"
        return metric_text(value) if value is not None else "采集中"

    host = measured(resources.get("cpus_used"), resources.get("cpu_status"))
    owned = measured(
        resources.get("owned_cpu_cores"), resources.get("owned_cpu_status")
    )
    throughput = summary.get("throughput", {})
    total = measured(summary.get("total_ticks_per_second"), throughput.get("status"))
    training = measured(
        summary.get("training_ticks_per_second"),
        "ok"
        if summary.get("training_ticks_per_second") is not None
        else throughput.get("status"),
    )
    lines = [
        f"主机可见 CPU 忙碌 {host}/{metric_text(resources.get('cpus_total'))} 核 · 本批次 {owned} 核 · 有效配额 {metric_text(resources.get('cpus_effective'))} 核 · 可用内存 {memory(resources.get('memory_available'))}",
        f"总吞吐 {total} ticks/s · 训练 {training} ticks/s（最近 {metric_text(throughput.get('window_seconds', 30))} 秒）",
    ]
    notes = []
    if resources.get("host_cpu_total") is not None and resources.get(
        "host_cpu_total"
    ) != resources.get("cpus_total"):
        notes.append(
            f"主机逻辑 CPU 总量 {metric_text(resources['host_cpu_total'])} 核；忙碌观测限于当前可见范围"
        )
    if throughput.get("reason"):
        reasons = {
            "warming execution window": "吞吐窗口采集中",
            "source progress is stale": "源进度已过期",
            "source has no execution counters": "源任务尚无执行计数",
            "legacy source has training-only counters": "旧快照仅有训练计数，执行总吞吐待新观测",
        }
        notes.append(reasons.get(throughput["reason"], "执行观测待补齐"))
    if resources.get("owned_cpu_reason"):
        notes.append(
            "批次 CPU "
            + (
                "采集中：新进程首样本"
                if resources["owned_cpu_reason"]
                == "owned process first CPU sample unavailable"
                else "观测不可用：拥有进程记录待核实"
            )
        )
    return [*lines, " · ".join(notes)]


def runtime_lines(summary, *, controlling=False, condensed=False):
    """Fixed-height control facts, separate from adaptive performance estimates."""
    state = summary.get("runtime_control", {})
    mode = state.get("mode", "manual")
    meaning = "手动目标" if mode == "manual" else "自适应上限"
    active, target = state.get("active", 0), state.get("limit")
    draining = target is not None and active > target
    rows = [
        f"控制 · {meaning} {shown(target)} · 实际运行 {shown(active)} · {state.get('status', 'unknown')}"
        + (" · 自然收缩中" if draining else ""),
        ("可控制 · +/- 调整" if controlling else "只读 monitor")
        + f" · {meaning} · CPU/内存预算保护（非操作系统硬隔离）",
        "阻塞原因 · " + (state.get("blocking_reason") or "无"),
        "队列 · " + ("排空后等待追加" if state.get("keep_open") else "排空自动结束"),
    ]
    return rows[:2] if condensed else rows


def control_footer(view, summary):
    if view.controlling and summary.get("runtime_control"):
        return "+/- 并发 · c 数值 · 空格 全局暂停/继续 · P 暂停选中 · R 恢复选中 · D 移除确认 · "
    return "c 修改总并发 · " if summary.get("manual") and view.controlling else ""


def manual_lines(summary, *, condensed=False):
    return [
        f"调度 · 手动上限 · 实际运行 {shown(summary.get('active'))}/{shown(summary.get('concurrency_limit'))} 个任务 · 自适应关闭",
        *(execution_lines(summary)[:2] if condensed else execution_lines(summary)),
        "队列开放 · 命令行 batch-add 追加任务"
        if summary.get("queue_open")
        else "队列已关闭，等待剩余任务结束",
    ]


def execution_card(summary):
    return Panel(
        Group(*(line(row, "cyan") for row in execution_lines(summary))),
        title=line("执行观测 · 当前阶段", "bold cyan"),
        border_style="cyan",
        padding=(0, 1),
    )


def summary_lines(summary):
    calibration = summary.get("calibration", {})
    measured, candidates = candidate_counts(calibration)
    rows = [
        f"调度阶段 {STAGES.get(summary.get('stage'), summary.get('stage', 'N/A'))}",
    ]
    if summary.get("runtime_control"):
        rows.extend(runtime_lines(summary))
    if summary.get("manual"):
        rows.extend(manual_lines(summary))
    elif summary.get("native_stage"):
        rows.extend(execution_lines(summary))
    elif online_mode(summary):
        rows.extend(performance_lines(summary, maximum=None))
        rows.append(resource_text(summary.get("resources", {})))
    else:
        rows.extend(
            [
                f"性能档 {shown(calibration.get('level', 'quick'))} · 混组调度 {shown(calibration.get('schedule_status', '未判定'))}",
                f"性能评估 {shown(calibration.get('wall_seconds'))}/{shown(calibration.get('limit_seconds'))}s；剩余 {shown(calibration.get('remaining_seconds'))}s；已测候选 {shown(measured)}/{shown(candidates) if candidates is not None else '未记录'}",
                (
                    "推荐设置 "
                    if calibration.get("calibrated", True)
                    else "起始设置（未经校准） "
                )
                + profile_summary(calibration.get("recommendation")),
                resource_text(summary.get("resources", {})),
            ]
        )
    if calibration.get("reason") and not online_mode(summary):
        rows.append("校准说明 " + compact(calibration["reason"]))
    for group, candidate in calibration.get("historical_candidates", {}).items():
        source = candidate.get("source", {})
        commit = source.get("git", {}).get("commit")
        versions = source.get("packages", {})
        rows.append(
            f"历史候选 {group}: {candidate.get('report')} · "
            f"源码 {commit[:12] if commit else '未知'} · "
            f"Ray {versions.get('ray')} / Torch {versions.get('torch')}；本次重测"
        )
    for entry in summary.get("entries", []):
        rows.append(
            f"{entry['experiment_id']}: {shown(entry.get('status'))} · {allocation_text(entry)}"
        )
        if entry.get("resource_change_reason") and entry.get("status") not in FINAL:
            rows.append("资源变更原因 " + entry["resource_change_reason"])
    return rows


def _online_groups(summary):
    groups = summary.get("online", {})

    def active_count(name, group):
        entries = [
            entry
            for entry in summary.get("entries", ())
            if entry.get("calibration_group") == name
        ]
        return (
            sum(entry.get("status") == "running" for entry in entries)
            if entries
            else group.get("active")
        )

    return sorted(
        (
            (name, {**group, "active": active_count(name, group)})
            for name, group in groups.items()
        ),
        key=lambda pair: (
            -(pair[1].get("active") or 0),
            pair[1].get("latest_throughput") is None,
        ),
    )


def performance_lines(summary, *, condensed=False, maximum=2):
    """Show fresh-run evidence, keeping admission limits distinct from workers."""
    ordered = _online_groups(summary)
    if not ordered:
        return ["在线调整 · 等待真实更新与内存观测；从 1 个实验开始"]
    rows = []
    reasons = {
        "warming up one real experiment": "预热，等待两个稳定窗口",
        "throughput windows unstable; holding concurrency": "吞吐波动，保持并发继续观测",
        "stable useful throughput; testing one more experiment": "吞吐稳定，尝试增加一个实验",
        "holding measured concurrency": "保持已确认并发",
        "less than 5% throughput gain; returning to measured concurrency": "增益不足 5%，回退至已确认并发",
    }
    maximum = 1 if condensed else maximum
    for name, group in ordered[:maximum]:
        prefix = "" if len(ordered) == 1 else f"组 {name[:6]} · "
        rows.append(
            f"{prefix}实际运行 {shown(group.get('active'))} · 在线并发上限 {shown(group.get('limit'))}/{shown(group.get('ceiling'))}"
            f" · 已确认 {shown(group.get('best_concurrency'))}"
        )
        rate = group.get("latest_throughput")
        rows.append(
            "最近完整窗口 "
            + (
                f"{metric_text(rate)} ticks/s（{shown(group.get('measured_concurrency'))} 个实验合计）"
                if rate is not None
                else "待测"
            )
        )
        if not condensed:
            reason = group.get("reason", "等待真实更新")
            rows.append(reasons.get(reason, reason))
    if maximum is not None and len(ordered) > maximum and not condensed:
        rows.append(f"另有 {len(ordered) - maximum} 个任务组 · 全部观测见 runtime.log")
    return rows


def performance_card(summary, *, condensed=False):
    return Panel(
        Group(
            *(
                line(row, "cyan")
                for row in performance_lines(summary, condensed=condensed)
            )
        ),
        title=line("性能 · 在线调整", "bold cyan"),
        border_style="cyan",
        padding=(0, 1),
    )


def _entry_rows(view):
    entries = (view.tuning or {}).get("entries", [])
    result = []
    by_id = {entry["experiment_id"]: entry for entry in entries}
    for row in view.tasks.values():
        if row.get("unit") == "entries":
            continue
        entry = by_id.pop(row["id"], None)
        if entry is not None or row.get("unit") in {
            "physical ticks",
            "environment steps",
            "ticks",
        }:
            result.append(
                (entry or dict(experiment_id=row["id"], status=row["status"]), row)
            )
    for entry in by_id.values():
        row = view.tasks.get(entry["experiment_id"])
        if row is None:
            row = {
                "id": entry["experiment_id"],
                "name": entry["experiment_id"],
                "status": entry.get("status", "unknown"),
                "stage": entry.get("status", "unknown"),
                "values": {},
                "learner": {},
            }
        result.append((entry, row))
    return result


def profile_summary(value):
    if isinstance(value, dict) and "threads" in value:
        return (
            f"线程 {shown(value.get('threads'))} · 环境 {shown(value.get('num_envs'))}"
            f" · 采样进程 {shown(value.get('sampling_processes'))}"
            f" · 实验并行 {shown(value.get('concurrency'))}"
            f" · {shown(value.get('device'))}"
        )
    if (
        isinstance(value, dict)
        and value
        and all(isinstance(item, dict) and "threads" in item for item in value.values())
    ):
        return "; ".join(
            f"{name[:6]}: {profile_summary(item)}" for name, item in value.items()
        )
    return compact(value)


def compact_trial_card(view, entry, row):
    """Keep an experiment card and its active case visible in short terminals."""
    if row["id"] not in view.tasks:
        return Panel(
            line("尚未启动 · 进度 N/A · ETA 待启动"), title=line(row["name"]), height=7
        )
    mode = (row.get("values", {}).get("workflow") or {}).get("mode")
    progress = bar(
        "评估案例" if mode == "evaluation" else "训练预算",
        row.get("completed"),
        row.get("total"),
    )
    resource = (
        "资源已释放"
        if entry.get("status") in FINAL
        else f"CPU 实际/请求 {shown(entry.get('actual_cpus'))}/{shown(entry.get('requested_cpus'))}"
    )
    detail = f"{shown(row.get('completed'))}/{shown(row.get('total'))} {row.get('unit', 'N/A')} · ETA {task_eta(view, row)}"
    if entry.get("pending_resize") and entry.get("status") not in FINAL:
        detail += " · 待生效"
    reason = row.get("reason") or (
        entry.get("resource_change_reason")
        if entry.get("status") not in FINAL
        else None
    )
    items = [
        progress,
        phase_bar(view, row),
        line(detail, "dim"),
        line(
            (
                row["reason"]
                if row.get("reason")
                else resource + (" · " + reason if reason else "")
            ),
            "bold red" if row.get("reason") else "yellow" if reason else "dim",
        ),
    ]
    return Panel(
        Group(*items),
        title=line(
            f"{row['name']} · {PHASE_NAMES.get(row.get('stage'), row.get('stage', 'N/A'))}",
            "bold",
        ),
        box=box.SQUARE,
        border_style="#74aaff",
        padding=(0, 1),
        height=7,
    )


def _recorded(value):
    if value is None or value == "":
        return "未记录"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def _gib(value):
    return "未记录" if value is None else f"{value / 1024**3:.2f} GiB"


def _task_resources(task, values):
    resources = dict(task.get("resources") or values.get("resources") or {})
    observed = resources.get("observed_at")
    if (
        resources.get("status") == "current"
        and isinstance(observed, (int, float))
        and time.time() - observed > 5
    ):
        resources.update(status="stale", cpu_cores=None, memory=None)
    return resources


def _validation_return(values, mode):
    record = (values.get("validation_results") or {}).get(mode) or {}
    value = record.get("mean_return")
    number = f"{value:.3g}" if isinstance(value, (float, int)) else "未记录"
    return number + (
        f"@u{record['update']}" if record.get("update") is not None else ""
    )


def runtime_feedback_commands(view):
    """Read observations plus this attachment's unacknowledged submissions."""
    commands = dict(getattr(view, "runtime_local_commands", {}))
    commands.update(
        (view.tuning or {}).get("runtime_control", {}).get("commands") or {}
    )
    return sorted(
        (
            row
            for row in commands.values()
            if (row.get("payload") or {}).get("kind") in {"pause", "resume", "remove"}
        ),
        key=lambda row: row.get("revision", row.get("received_at", 0)),
    )


def pending_pause(view, task_id=None):
    candidates = [
        row
        for row in runtime_feedback_commands(view)
        if (row.get("payload") or {}).get("task_id") == task_id
    ]
    if not candidates:
        return None
    latest = candidates[-1]
    if (
        latest.get("payload", {}).get("kind") == "pause"
        and latest.get("status") != "rejected"
        and (
            latest.get("status") == "received"
            or (latest.get("progress") or {}).get("state") in {None, "waiting"}
        )
    ):
        return latest
    return None


def runtime_feedback(view):
    commands = runtime_feedback_commands(view)
    chosen = getattr(view, "runtime_feedback_id", None)
    selected = next(
        (row for row in commands if row.get("id", row.get("command_id")) == chosen),
        None,
    )
    if selected:
        return selected
    waiting = [
        row
        for row in commands
        if (row.get("progress") or {}).get("state") == "waiting"
        or row.get("status") == "received"
    ]
    return (waiting or commands)[-1] if commands else None


def runtime_feedback_lines(view, fallback="无操作请求"):
    receipt = runtime_feedback(view)
    if not receipt:
        return ["操作反馈 · 暂无请求", ""]
    payload = receipt.get("payload") or {}
    progress = receipt.get("progress") or {}
    action = payload.get("kind", "操作")
    target = payload.get("task_id")
    scope = "全局" if target is None else "任务 " + str(target)
    status = receipt.get("status")
    state = progress.get("state")
    waiting = state == "waiting"
    label = {
        "completed": "实际请求已落实",
        "failed": "失败",
        "superseded": "已被后续请求替代",
    }.get(state)
    if status == "rejected":
        label = "请求被拒绝"
    elif waiting:
        label = {
            "pause": "正在安全暂停",
            "resume": "等待实际恢复",
            "remove": "等待实际移除",
        }.get(action, "等待观测")
    elif not label:
        label = (
            "已收到 · 尚未应用" if status == "received" else "意图已应用 · 实际状态未知"
        )
    elapsed = progress.get("elapsed_seconds")
    if (
        elapsed is None
        and status == "received"
        and receipt.get("received_at") is not None
    ):
        elapsed = max(0, time.time() - receipt["received_at"])
    if elapsed is not None and waiting and progress.get("updated_at") is not None:
        elapsed += max(0, time.time() - progress["updated_at"])
    elapsed_text = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else "未知"
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 2) % 10] if waiting else ""
    targets = progress.get("targets") or []
    tasks = progress.get("tasks") or {}
    paused = sum((tasks.get(key) or {}).get("status") == "paused" for key in targets)
    failed = progress.get(
        "failed_count",
        sum(
            (tasks.get(key) or {}).get("status")
            in {"failed", "force_stopped", "interrupted", "stopped", "removed"}
            for key in targets
        ),
    )
    pending = progress.get(
        "pending_count",
        sum(
            (tasks.get(key) or {}).get("status")
            not in {
                "paused",
                "completed",
                "removed",
                "stopped",
                "failed",
                "force_stopped",
                "interrupted",
            }
            for key in targets
        ),
    )
    if (
        progress.get("total_count") is not None
        and progress.get("settled_count") is not None
    ):
        pending = max(0, progress["total_count"] - progress["settled_count"])
    if waiting:
        phases = {
            tasks[key].get("phase")
            for key in targets
            if key in tasks
            and tasks[key].get("status")
            not in {
                "paused",
                "completed",
                "failed",
                "stopped",
                "removed",
                "interrupted",
                "force_stopped",
            }
        }
        if action == "pause" and len(phases) == 1:
            label = {
                "saving": "正在保存",
                "waiting_exit": "等待进程退出",
                "waiting_boundary": "等待安全边界",
                "safe_boundary": "等待安全边界",
            }.get(next(iter(phases)), label)
        if failed:
            label += " · 部分失败"
    counts = (
        f"已暂停{paused}/{len(targets)} 待完成{pending} 失败{failed}"
        if action == "pause" and targets
        else "状态以实际观测为准"
    )
    # Scope is abbreviated only here; complete IDs and every target remain in details.
    scope = scope if len(scope) <= 20 else scope[:17] + "…"
    action_label = {"pause": "暂停", "resume": "恢复", "remove": "移除"}.get(
        action, action
    )
    timing = (
        f"已用{elapsed_text} · 剩余未知"
        if waiting or status == "received"
        else f"用时{elapsed_text}"
    )
    return [f"{action_label} {scope} · {label} {spinner}", f"{counts} · {timing}"]


def runtime_feedback_detail(view):
    receipt = runtime_feedback(view)
    if not receipt:
        return []
    progress = receipt.get("progress") or {}
    payload = receipt.get("payload") or {}
    lines = [
        "操作范围 " + str(payload.get("task_id") or "全局"),
        "命令 " + str(receipt.get("id")),
        "状态 " + str(receipt.get("status")),
        "结果 "
        + str(
            receipt.get("reason")
            or progress.get("state")
            or receipt.get("result")
            or "未记录"
        ),
    ]
    targets = progress.get("targets") or []
    natural = sum(
        ((progress.get("tasks") or {}).get(key) or {}).get("status") == "completed"
        for key in targets
    )
    lines.append(f"冻结目标 {len(targets)} · 自然完成 {natural}（非已暂停）")
    for target in targets:
        task = (progress.get("tasks") or {}).get(target) or {}
        lines.append(
            f"  {target}: {_recorded(task.get('status'))} · 阶段 {_recorded(task.get('phase'))} · {_recorded(task.get('reason'))}"
        )
    return lines


def runtime_status(status):
    return {
        "running": "运行中",
        "active": "运行中",
        "queued": "排队中",
        "pending": "排队中",
        "waiting_resources": "等待资源",
        "waiting": "等待中",
        "paused": "已暂停",
        "pausing": "暂停中",
        "completed": "已完成",
        "failed": "失败",
        "removed": "已移除",
        "removing": "移除中",
        "stopped": "已停止",
        "stopping": "停止中",
        "interrupted": "已中断",
        "force_stopped": "强制停止",
        "draining": "自然收缩中",
    }.get(status, status or "未知")


def runtime_document(
    view,
    *,
    title,
    content,
    sticky=(),
    footer="",
    compact_footer=None,
    offset_name="runtime_detail_offset",
    feedback=True,
):
    """Keep target identity and commands outside the scrollable evidence body."""
    from rich.text import Text

    width = max(1, view.console.width - 4)
    text = Text("\n").join(
        value if isinstance(value, Text) else Text(str(value)) for value in content
    )
    lines = text.wrap(view.console, width, overflow="fold")
    top = [
        value
        if isinstance(value, Text)
        else Text(str(value), style="bold", overflow="fold")
        for value in sticky
    ]
    if feedback and runtime_feedback(view):
        top.extend(line(value) for value in runtime_feedback_lines(view) if value)
    notice = line(str(view.notice or ""), "yellow")
    footer_text = Text(footer, overflow="fold")
    if view.notice:
        top.append(notice)

    def available():
        overhead = (
            len(
                view.console.render_lines(
                    Group(*top, Text("页码"), footer_text),
                    view.console.options.update(width=width, height=None),
                    pad=False,
                )
            )
            + 2
        )
        return view.console.height - 1 - overhead

    if available() < 3 and compact_footer:
        footer_text = Text(compact_footer, overflow="fold")
    if available() < 3 and feedback:
        # Reserve the failure evidence and navigation before optional feedback.
        top = top[: len(sticky)] + ([notice] if view.notice else [])
    capacity = max(1, available())
    offset = min(max(0, getattr(view, offset_name, 0)), max(0, len(lines) - capacity))
    setattr(view, offset_name, offset)
    body = list(lines[offset : offset + capacity])
    if (
        len(body) > 1
        and body[-1].get_style_at_offset(view.console, 0)
        == view.console.get_style("bold cyan")
        and offset + capacity < len(lines)
    ):
        # Keep a section heading with its first value on the next page.
        body[-1] = Text("")
        capacity -= 1
    setattr(view, offset_name + "_page_size", capacity)
    return Panel(
        Group(
            *top,
            *body,
            Text(
                f"详情 {offset + 1}–{min(offset + capacity, len(lines))}/{len(lines)}"
            ),
            footer_text,
        ),
        title=title,
        subtitle=line(getattr(view, "observation_notice", None) or ""),
        height=max(3, view.console.height - 1),
        padding=(0, 1),
    )


def runtime_help(view):
    content = [
        "浏览（不发送控制命令）",
        "↑/↓ 或 j/k 选择任务；←/→ 或 [/] 翻页",
        "Enter 或 i 打开任务详情；Esc 返回同一任务",
        "# 为固定任务序号；! 表示异常或资源观测过期，详情可查看原因",
        "总进度包括训练、验证与评估；ETA 的 * 表示恢复/资源就绪后的估计",
        "/ 搜索任务 ID/名称；Enter 应用；Esc 取消；空查询清除搜索",
        "f 循环未完成/失败、全部、历史；e 切换仅失败",
        "n 定位下一失败任务（进入仅失败视图，清除搜索）",
        "a 查看下一条操作记录；d 离开界面",
        "详情：↑/↓ 逐行；PgUp/PgDn 翻页；g/G 首尾；r 原始快照",
    ]
    if view.controlling:
        content += [
            "调度控制（仅 attach）",
            "+/- 并发：提交相对变化；c 数值：输入目标并发",
            "空格 全局暂停/继续；个别任务的暂停保持不变",
            "P 暂停选中；R 恢复选中；恢复仍需等待并发和资源",
            "Q 优先队列：↑/↓ 选任务；Enter 加入/取消；←/→ 前移/后移；Esc 返回",
            "优先队列只影响未启动任务；资源与依赖满足后按指定顺序启动",
            "D 移除确认：仅列表可用，输入目标短 ID；保留文件",
            "Ctrl+C×2 停止批次：3 秒内再次按下；单次不是停止",
            "任务自动离开当前筛选后，必须重新选择再控制",
            "模式、资源预算、追加任务及关闭队列使用 batch-* CLI",
        ]
    else:
        content += [
            "只读监控：不发送暂停、恢复、移除或并发变更",
            "Ctrl+C×2 关闭监控；不会停止批次",
        ]
    return runtime_document(
        view,
        title="快捷键帮助",
        content=content,
        feedback=False,
        offset_name="runtime_help_offset",
        footer="↑/↓ 滚动 · PgUp/PgDn 翻页 · Esc/? 返回 · d 离开",
    )


def runtime_detail(view, row, task):
    from rich.cells import cell_len
    from rich.text import Text

    from smartsom.telemetry.estimates import task_work_progress

    values, learner = row.get("values") or {}, row.get("learner") or {}
    plan = values.get("workflow") or view.workflow or {}
    state = (view.tuning or {}).get("runtime_control", {})
    raw = getattr(view, "runtime_detail_raw", False)
    reason = row.get("reason") or task.get("reason") or values.get("warning")
    content = []

    def section(label):
        width = max(1, view.console.width - 4)
        heading = Text(label + " ", style="bold cyan")
        heading.append("─" * max(0, width - cell_len(label) - 1), "dim")
        content.append(heading)

    def field(label, value, style=""):
        # Align by terminal cells, including Chinese labels; wrapped values
        # retain their full evidence in the scrollable document.
        prefix = Text().append(label, "dim")
        prefix.pad_right(max(1, 16 - cell_len(label)))
        value_text = value if isinstance(value, Text) else Text(str(value), style=style)
        width = max(1, view.console.width - 4)
        if prefix.cell_len >= width // 2 and cell_len(label) >= 16:
            content.append(Text(label, style="dim"))
            prefix = Text(" " * 16)
        wrapped = value_text.wrap(
            view.console, max(1, width - prefix.cell_len), overflow="fold"
        )
        for index, value_line in enumerate(wrapped):
            content.append(
                (
                    prefix.copy() if index == 0 else Text(" " * prefix.cell_len)
                ).append_text(value_line)
            )

    if raw:
        content = json.dumps(
            {
                "task": row,
                "runtime_task": task,
                "budget": state.get("budget"),
                "commands": state.get("commands"),
                "batch_resources": (view.tuning or {}).get("resources"),
                "batch_overview": view.overview,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ).splitlines()
    else:
        if reason:
            section("失败 / 阻塞原因")
            content.append(Text(str(reason), style="bold red"))
        progress = task_work_progress(row, plan, row.get("execution"))
        phase, phase_fraction, stages = runtime_phase(row)
        section("运行进度")
        field(
            "任务进度",
            runtime_progress(
                progress["fraction"], cells=18 if view.console.width >= 100 else 10
            ),
        )
        field(
            "剩余时间",
            "ETA "
            + runtime_eta(view, row)
            + (
                " · 含排队等待"
                if row.get("status")
                in {"pending", "queued", "waiting", "waiting_resources"}
                else ""
            ),
        )
        field("执行阶段", stages)
        field("当前阶段", runtime_phase_text(phase, phase_fraction))
        _, phase_done, phase_total, _, _ = _runtime_phase_facts(row)
        if row.get("stage") in {"sampling", "training", "running"}:
            phase_done, phase_total = row.get("completed"), row.get("total")
        if phase_fraction is not None and phase_done is not None and phase_total:
            phase_units = " " + row.get("unit", "") if phase == "训练" else ""
            field("阶段完成", f"{phase_done} / {phase_total}{phase_units}")
        if phase != "训练" or phase_fraction is None:
            field(
                "训练累计",
                f"{_recorded(row.get('completed'))}/{_recorded(row.get('total'))} {row.get('unit', '')}",
            )
        resources = _task_resources(task, values)
        observed = resources.get("observed_at")
        age = (
            f"{max(0, time.time() - observed):.0f} 秒前更新"
            if isinstance(observed, (int, float))
            else "未记录"
        )
        section("资源")
        field(
            "当前占用",
            f"CPU {_recorded(resources.get('cpu_cores'))} 核 · RAM {_gib(resources.get('memory'))}",
        )
        field("已采样内存峰值", _gib(resources.get("memory_peak")))
        field(
            "资源观测",
            age + (" · 已过期" if resources.get("status") == "stale" else ""),
        )
        section("检查点与日志")
        field("已保存路径", _recorded(values.get("checkpoint_path")))
        field("保存轮次", "update " + _recorded(values.get("checkpoint_update")))
        field("恢复验证", "未记录（保存事件不是恢复验证）")
        field("日志路径", _recorded(values.get("log_path", task.get("log_path"))))
        section("验证回报（不代表唯一选模目标）")
        for mode in ("argmax", "sampling"):
            result = (values.get("validation_results") or {}).get(mode) or {}
            field(
                mode + ":",
                f"{_validation_return(values, mode)} · 样本 {_recorded(result.get('return_samples'))}",
            )
        field(
            "验证轮次",
            f"{_recorded(values.get('validation_round'))} · 已完成批次 {_recorded(values.get('validation_batches_finished'))}",
        )
        algorithm = str(values.get("algorithm", plan.get("algorithm", ""))).upper()
        for heading, names, relevant in (
            (
                "PPO 策略 / 价值 / 熵 / KL",
                {
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "entropy_loss",
                    "approx_kl",
                    "mean_kl_loss",
                },
                "PPO" in algorithm,
            ),
            ("DQN / 通用学习损失", {"loss", "total_loss"}, "DQN" in algorithm),
        ):
            selected = {
                k: v for k, v in learner.items() if k.rsplit("/", 1)[-1] in names
            }
            if selected or relevant:
                section(heading)
                if selected:
                    for key, value in selected.items():
                        field(key, _recorded(value))
                else:
                    field("观测值", "未记录")
        diagnostics = (values.get("learner_diagnostics") or {}).get("groups") or {}
        for group, facts in diagnostics.items():
            section("策略组 " + group + " · 累计加权诊断")
            for name, metric in (facts.get("metrics") or {}).items():
                field(
                    name,
                    f"{_recorded(metric.get('mean'))} / weight {_recorded(metric.get('weight'))} · 缺失批 {_recorded(metric.get('unavailable_minibatches'))}",
                )
            for key in (
                "value_explained_variance",
                "observed_minibatches",
                "optimization_steps",
                "replay_length",
                "replay_capacity",
                "batch_size",
                "target_clock",
            ):
                if key in facts:
                    field(key, _recorded(facts[key]))
        section("阶段耗时 / 工作量 / 速率")
        phases = (row.get("execution") or {}).get("phases") or {}
        for name, facts in phases.items():
            field(
                {
                    "training": "训练",
                    "validation": "验证",
                    "evaluation": "评估",
                    "loading": "准备",
                    "finalization": "收尾",
                }.get(name, PHASE_NAMES.get(name, name)),
                f"{duration(facts.get('seconds')) if facts.get('seconds') is not None else '未记录'} · work {_recorded(facts.get('work'))} · rate {_recorded(facts.get('rate'))}",
            )
        if not phases:
            field("阶段观测", "未记录")
        batch_resources = (view.tuning or {}).get("resources") or {}
        overview = view.overview or {}
        section("批次概况")
        field(
            "主机可用",
            f"CPU {_recorded(batch_resources.get('cpus_available'))} · RAM {_gib(batch_resources.get('memory_available'))}",
        )
        field(
            "工作流",
            f"{_recorded(overview.get('work_completed'))}/{_recorded(overview.get('work_total'))} · ETA {runtime_eta(view)}",
        )
        field("调度等待原因", str(state.get("blocking_reason") or "无"))
        section("进度与 ETA 说明")
        content.append(
            Text(
                "任务进度按训练、验证和评估的计划工作量计算；阶段百分比只表示当前阶段，切换阶段后重新计数。* 表示恢复或资源可用后的条件估计。",
                style="dim",
            )
        )
        section("操作记录")
        content.extend(runtime_feedback_detail(view))
        field("逻辑预算", json.dumps(state.get("budget"), ensure_ascii=False))
    identity = Text("任务 " + row["id"], style="bold")
    identity.append(
        "  ·  状态 " + runtime_status(row.get("status")),
        "red" if row.get("status") == "failed" else "cyan",
    )
    metadata = Text(
        f"{_recorded(values.get('algorithm', plan.get('algorithm')))} · seed {_recorded(plan.get('training_seed'))} · update {_recorded(values.get('updates', values.get('ppo_updates', values.get('learner_updates'))))}/{_recorded(plan.get('round_total'))}",
        style="dim",
    )
    return runtime_document(
        view,
        title="实验详情 · " + ("原始快照" if raw else "概览"),
        content=content,
        sticky=[identity, metadata],
        footer="↑↓ 滚动 · PgUp/PgDn 翻页 · r 原始 · Esc 返回\n"
        + ("P 暂停 · R 恢复 · " if view.controlling else "")
        + "g/G 首尾 · a 记录 · ? 帮助 · d 离开",
        compact_footer="↑/↓ 滚动 · Esc 返回 · ? 帮助",
    )


def _runtime_phase_facts(row):
    """Keep current phase facts distinct from the cumulative training budget."""
    from smartsom.telemetry.estimates import phase_of

    values = row.get("values") or {}
    stage = row.get("stage")
    phase = phase_of(row)
    label = "阶段 " + PHASE_NAMES.get(stage, _recorded(stage))
    done = total = speed = None
    unit = None
    evaluation_phases = {"validation", "evaluation", "initial", "rule", "random"}
    if phase in evaluation_phases:
        prefix = "validation" if phase == "validation" else "evaluation"
        label = "阶段 " + PHASE_NAMES.get(phase, phase)
        done, total = (
            values.get(prefix + "_finished"),
            values.get(prefix + "_requested"),
        )
        unit = "ticks/s"
    elif phase == "training" and stage in {"training", "running", "sampling"}:
        label = "采样·训练累计" if stage == "sampling" else "训练累计"
        done, total = row.get("completed"), row.get("total")
        unit = {
            "physical ticks": "ticks/s",
            "environment steps": "steps/s",
            "adapter decisions": "decisions/s",
            "sampling decisions": "decisions/s",
        }.get(row.get("unit"))
    # The producer's training rate includes sampling AND optimization. It is not
    # a rate for either subphase; no optimizer denominator is currently published.
    if (
        unit
        and stage not in {"sampling", "updating", "optimizing", "learning_metrics"}
        and row.get("status") not in FINAL
    ):
        speed = (
            ((row.get("execution") or {}).get("phases") or {}).get(phase) or {}
        ).get("rate")
    return label, done, total, speed, unit


def runtime_eta(view, row=None):
    from smartsom.telemetry.estimates import coarse_time

    state = (view.tuning or {}).get("runtime_control", {})
    if row and row.get("status") in FINAL:
        return "已结束"
    if getattr(view, "observation_notice", None):
        return "观测过期"
    execution = view.execution or {}
    estimate = (
        execution.get("tasks", {}).get(row["id"]) if row else execution.get("batch")
    ) or {}
    if not row and not estimate:
        estimate = {"seconds": (view.overview or {}).get("eta_seconds")}
    if row and not estimate:
        estimate = (row.get("execution") or {}).get("estimate") or {}
    if state.get("paused") or row and row.get("status") == "paused":
        return "待恢复"
    if estimate.get("conditional"):
        conditional = (
            estimate.get("conditional_completion_seconds")
            if row
            else estimate.get("conditional_seconds")
        )
        return coarse_time(conditional) + "*" if conditional is not None else "待资源"
    seconds = estimate.get("completion_seconds") if row else estimate.get("seconds")
    if (
        seconds is None
        and row
        and row.get("status") in {"queued", "pending", "waiting_resources"}
        and estimate.get("conditional_completion_seconds") is not None
    ):
        return coarse_time(estimate["conditional_completion_seconds"]) + "*"
    if (
        seconds is None
        and row
        and row.get("status") not in {"queued", "pending", "waiting_resources"}
    ):
        seconds = estimate.get("seconds")
    return coarse_time(seconds) if seconds is not None else "估算中"


def runtime_phase(row):
    from rich.text import Text

    from smartsom.telemetry.estimates import phase_of

    phase = phase_of(dict(row, status="running"))
    label, done, total, _, _ = _runtime_phase_facts(dict(row, status="running"))
    if phase == "training" and row.get("stage") in {"sampling", "training", "running"}:
        label, done, total = "训练", row.get("completed"), row.get("total")
    else:
        label = {
            "loading": "准备",
            "initial": "基线",
            "rule": "规则",
            "random": "随机",
            "finalization": "收尾",
        }.get(phase, label.removeprefix("阶段 "))
    fraction = min(1, max(0, done / total)) if done is not None and total else None
    if row.get("status") in {"queued", "pending", "waiting_resources"}:
        label, fraction, phase = "待启动", None, None
    elif row.get("status") == "completed":
        label, fraction, phase = "完成", 1, None
    chips = Text()
    for index, (key, name) in enumerate(
        [
            ("loading", "准备"),
            ("training", "训练"),
            ("validation", "验证"),
            ("evaluation", "评估"),
            ("saving", "保存"),
        ]
    ):
        if index:
            chips.append(" › ", "dim")
        active = (
            phase == key
            or key == "evaluation"
            and phase in {"initial", "rule", "random"}
            or key == "saving"
            and phase == "finalization"
        )
        chips.append(name, "bold cyan underline" if active else "dim")
    return label, fraction, chips


def runtime_phase_text(label, fraction, *, narrow=False):
    """Keep the phase percentage subordinate to its label and out of the bar."""
    from rich.text import Text

    value = Text(label)
    if fraction is not None:
        separator = "" if narrow else " · "
        value.append(separator + f"{int(min(1, max(0, fraction)) * 100):3}%", "dim")
    return value


def runtime_render(view):
    """Stable experiment cards and all-batch facts from the published snapshot."""
    if getattr(view, "runtime_help", False):
        return runtime_help(view)
    if getattr(view, "control_input", None) == "remove":
        # Confirmation owns the viewport; background cards cannot obscure the target.
        notice = view.notice
        view.notice = None
        try:
            return runtime_document(
                view,
                title="确认移除任务",
                content=str(notice or "").splitlines(),
                feedback=False,
                offset_name="runtime_confirmation_offset",
                footer="↑/↓ 滚动 · 输入目标短 ID\nEnter 确认移除 · Esc/Ctrl+C 取消",
            )
        finally:
            view.notice = notice
    summary = view.tuning or {}
    state = summary["runtime_control"]
    tasks = state.get("tasks", {})
    rows = dict(view.tasks)
    for key, task in tasks.items():
        rows.setdefault(key, dict(id=key, name=key, values={}))
    aggregates = {
        task["file_id"]
        for key, task in tasks.items()
        if task.get("file_id") and task["file_id"] != key
    }
    canonical = []
    for key, row in rows.items():
        if key not in aggregates:
            canonical.append(
                (
                    {},
                    {
                        **row,
                        "status": tasks.get(key, {}).get(
                            "status", row.get("status", "unknown")
                        ),
                    },
                )
            )
    counts = {}
    for _, row in canonical:
        status = {
            "pending": "queued",
            "waiting_resources": "queued",
            "waiting": "queued",
            "active": "running",
        }.get(row["status"], row["status"])
        counts[status] = counts.get(status, 0) + 1
    old_anchor = view.task_pages.anchor
    visible = view.task_pages.filtered(canonical, paged=view.paging)
    if getattr(view, "runtime_queue", False):
        priority = {
            key: index for index, key in enumerate(state.get("priority_tasks", []))
        }
        visible.sort(key=lambda pair: priority.get(pair[1]["id"], len(priority)))
    view.selection_notice = (
        "原选中任务已离开当前视图；请重新选择后操作"
        if old_anchor and old_anchor != view.task_pages.anchor
        else getattr(view, "selection_notice", None)
    )
    if visible and view.task_pages.anchor not in {row["id"] for _, row in visible}:
        view.task_pages.anchor = visible[0][1]["id"]
    detail_id = getattr(view, "runtime_detail_task", None)
    if detail_id and not getattr(view, "control_input", None):
        row = next(
            (row for _, row in canonical if row["id"] == detail_id), {"id": detail_id}
        )
        return runtime_detail(view, row, tasks.get(detail_id, {}))
    resources = summary.get("resources") or {}
    overview = view.overview or {}
    active, limit = state.get("active"), state.get("limit")
    draining = (
        isinstance(active, (int, float))
        and isinstance(limit, (int, float))
        and active > limit
    )
    reason = state.get("blocking_reason") or (
        "自然收缩 draining"
        if draining
        else "已暂停"
        if state.get("paused")
        else "已满"
        if active is not None and limit is not None and active == limit
        else "未记录"
    )
    training_rows = [
        row
        for _, row in canonical
        if row.get("total") is not None
        and row.get("unit")
        in {
            "physical ticks",
            "environment steps",
            "adapter decisions",
            "sampling decisions",
        }
    ]
    units = {row.get("unit") for row in training_rows}
    training = (
        f"{sum(row['completed'] for row in training_rows)}/{sum(row['total'] for row in training_rows)} {next(iter(units)) or ''}"
        if training_rows
        and len(units) == 1
        and len(training_rows) == len(canonical)
        and all(row.get("completed") is not None for row in training_rows)
        else "未记录（预算未知或单位不同）"
    )
    eta_text = runtime_eta(view)
    header = [
        line(
            f"完成 {counts.get('completed', 0)}/{len(canonical)} 运行 {counts.get('running', 0)} 排队 {counts.get('queued', 0)} 暂停 {counts.get('paused', 0)} 暂停中 {counts.get('pausing', 0)} 失败 {counts.get('failed', 0)} 已移除 {counts.get('removed', 0)}",
            "bold",
        ),
        line(
            f"目标 {shown(limit)} / 实际 {shown(active)} · "
            + {"manual": "手动", "adaptive": "自适应"}.get(state.get("mode"), "未知")
            + (" · draining" if draining else ""),
            "cyan",
        ),
        line(
            f"批次 CPU {_recorded(resources.get('owned_cpu_cores'))} 核 · RAM {_gib(resources.get('owned_memory_bytes'))}"
        ),
        line(
            f"主机可用 CPU {_recorded(resources.get('cpus_available'))} · RAM {_gib(resources.get('memory_available'))}"
        ),
        line("训练 " + training),
        line(
            f"工作流 {_recorded(overview.get('work_completed'))}/{_recorded(overview.get('work_total'))} · ETA {eta_text}"
        ),
    ]
    from rich.text import Text

    from smartsom.telemetry.estimates import task_work_progress

    progress = {
        row["id"]: task_work_progress(
            row,
            (row.get("values") or {}).get("workflow") or view.workflow or {},
            row.get("execution"),
        )
        for _, row in canonical
    }
    total_work = sum(value["total"] or 0 for value in progress.values())
    known_work = progress and all(
        value["total"] is not None for value in progress.values()
    )
    batch_fraction = (
        sum(value["completed"] or 0 for value in progress.values()) / total_work
        if known_work and total_work
        else None
    )
    if batch_fraction is not None and counts.get("completed", 0) != len(canonical):
        batch_fraction = min(0.99, batch_fraction)
    batch_progress = Text("批次进度 ", no_wrap=True, overflow="ellipsis")
    batch_progress.append_text(
        runtime_progress(batch_fraction, cells=18 if view.console.width >= 100 else 10)
    )
    batch_progress.append(" │ ETA " + runtime_eta(view), "cyan")
    # Reserve digit slots from the batch size, not the current counts. A 9→10
    # transition must not move any following label or the task table.
    digits = max(2, len(str(len(canonical))))
    compact = view.console.width < 76
    status_line = Text(no_wrap=True, overflow="ellipsis")
    for index, (key, label) in enumerate(
        [
            ("completed", "完成"),
            ("running", "运行"),
            ("queued", "排队"),
            ("paused", "暂停"),
            ("pausing", "暂停中"),
            ("failed", "失败"),
            ("removed", "已移除"),
        ]
    ):
        if index:
            status_line.append(" " if compact else "  ")
        status_line.append(label + ("" if compact else " "), "dim")
        status_line.append(f"{counts.get(key, 0):>{digits}}", "bold")
        if key == "completed":
            status_line.append(f"/{len(canonical):>{digits}}", "dim")

    def fixed_value(value, width):
        result = Text(str(value), no_wrap=True, overflow="ellipsis")
        result.truncate(width, overflow="ellipsis")
        result.align("right", width)
        return result

    resource_line = Text("批次 CPU ", no_wrap=True, overflow="ellipsis")
    resource_line.append_text(
        fixed_value(
            _recorded(resources.get("owned_cpu_cores"))
            if resources.get("owned_cpu_cores") is not None
            else "—",
            5,
        )
    )
    resource_line.append(" 核 │ RAM ", "dim")
    resource_line.append_text(
        fixed_value(
            _gib(resources["owned_memory_bytes"])
            if resources.get("owned_memory_bytes") is not None
            else "—",
            10,
        )
    )
    resource_line.append(" │ 并发 ", "dim")
    resource_line.append_text(fixed_value(shown(active), digits))
    resource_line.append("/", "dim")
    resource_line.append_text(fixed_value(shown(limit), digits))
    resource_line.append(
        " " + {"manual": "手动", "adaptive": "自适应"}.get(state.get("mode"), "未知"),
        "cyan",
    )
    short_header = [status_line, batch_progress, resource_line]
    cells, table_rows = [], []
    ordinals = {key: index + 1 for index, key in enumerate(view.task_pages.order)}
    for _, row in visible:
        values = row.get("values") or {}
        plan = values.get("workflow") or view.workflow or {}
        observations = _task_resources(tasks.get(row["id"], {}), values)
        stage_label, stage_done, stage_total, speed, speed_unit = _runtime_phase_facts(
            row
        )
        elapsed = values.get("elapsed_seconds")
        if (
            elapsed is None
            and row.get("started_at") is not None
            and row.get("updated_at") is not None
        ):
            elapsed = max(0, row["updated_at"] - row["started_at"])
        elapsed_text = duration(elapsed) if elapsed is not None else "未记录"
        speed_text = _recorded(speed) + (
            " " + speed_unit if speed is not None and speed_unit else ""
        )
        cells.append(
            Panel(
                Group(
                    line(
                        f"{_recorded(values.get('algorithm', plan.get('algorithm')))} · seed {_recorded(plan.get('training_seed'))} · {runtime_status(row['status'])}"
                    ),
                    line(
                        f"update {_recorded(values.get('updates', values.get('ppo_updates', values.get('learner_updates'))))}/{_recorded(plan.get('round_total'))} · 预算 {_recorded(row.get('completed'))}/{_recorded(row.get('total'))} · 耗时 {elapsed_text}"
                    ),
                    bar(
                        f"{stage_label} · 轮 {_recorded(values.get('validation_round'))}",
                        stage_done,
                        stage_total,
                        short=True,
                    )
                    if stage_done is not None
                    and stage_total is not None
                    and stage_total > 0
                    else line(
                        f"{stage_label} · 进度未记录 · 验证轮 {_recorded(values.get('validation_round'))}"
                    ),
                    line(
                        f"速度 {speed_text} · CPU {_recorded(observations.get('cpu_cores'))} 核 · RAM {_gib(observations.get('memory'))}"
                    ),
                    line(
                        f"return argmax {_validation_return(values, 'argmax')} · sampling {_validation_return(values, 'sampling')}"
                    ),
                    line(
                        str(
                            values.get("warning")
                            or row.get("reason")
                            or (
                                "资源观测已过期"
                                if observations.get("status") == "stale"
                                else ""
                            )
                        ),
                        "yellow",
                    ),
                ),
                title=TaskIdentity(view, row["id"]),
                height=8,
                border_style="red" if row["status"] == "failed" else "",
                padding=(0, 1),
            )
        )
        warning = (
            row.get("reason")
            or values.get("warning")
            or tasks.get(row["id"], {}).get("reason")
            or ("资源观测已过期" if observations.get("status") == "stale" else "")
        )
        status = row["status"]
        pending = status in {"pending", "queued", "waiting", "waiting_resources"}
        stage = (
            "采样/累计"
            if stage_label == "采样·训练累计"
            else "训练累计"
            if stage_label == "训练累计"
            else stage_label.removeprefix("阶段 ")
        )
        if view.console.width < 56 and stage_label in {"采样·训练累计", "训练累计"}:
            stage = "累计"
        fraction = (
            f"{min(1, max(0, stage_done / stage_total)):.0%}"
            if stage_done is not None and stage_total and stage_total > 0
            else "—"
        )
        progress_text = (
            f"{stage_label} {stage_done}/{stage_total}"
            if stage_done is not None and stage_total
            else f"{stage_label} · 进度未记录"
        )
        table_rows.append(
            {
                "id": row["id"],
                "name": row.get("name") or row["id"],
                "number": ordinals[row["id"]],
                "status": runtime_status(status) + ("!" if warning else ""),
                "status_style": "red"
                if status in {"failed", "force_stopped", "interrupted"}
                else "green"
                if status == "completed"
                else "yellow"
                if status in {"paused", "pausing"}
                else "cyan"
                if status in {"running", "active"}
                else "",
                "experiment": f"{_recorded(values.get('algorithm', plan.get('algorithm')))}/{_recorded(plan.get('training_seed'))}",
                "algorithm": _recorded(values.get("algorithm", plan.get("algorithm"))),
                "seed": _recorded(plan.get("training_seed")),
                "fraction": progress[row["id"]]["fraction"],
                "eta": runtime_eta(view, row),
                "speed": speed_text,
                "progress": "待启动"
                if pending
                else "已完成"
                if status == "completed"
                else f"{stage}{fraction}",
                "cpu": _recorded(observations.get("cpu_cores"))
                if observations.get("cpu_cores") is not None
                else "—",
                "memory": f"{observations['memory'] / 1024**3:.2f}"
                if observations.get("memory") is not None
                else "—",
                "elapsed": elapsed_text,
                "warning": str(warning),
                "context": str(warning)
                if warning
                else f"{progress_text} · 速度 {speed_text}",
            }
        )
        phase_label, phase_fraction, _ = runtime_phase(row)
        rank = (
            state.get("priority_tasks", []).index(row["id"]) + 1
            if row["id"] in state.get("priority_tasks", [])
            else None
        )
        if rank and pending:
            phase_label = f"优先#{rank}"
        table_rows[-1].update(
            phase_compact=runtime_phase_text(phase_label, phase_fraction),
            phase_narrow=runtime_phase_text(phase_label, phase_fraction, narrow=True),
            priority=rank,
        )
    receipt = runtime_feedback(view) or {}
    feedback_style = (
        "yellow"
        if receipt.get("status") == "rejected"
        or (receipt.get("progress") or {}).get("state") in {"waiting", "failed"}
        else ""
    )
    return render_runtime_panel(
        view,
        header=header,
        short_header=short_header,
        ids=[row["id"] for _, row in visible],
        cards=cells,
        table_rows=table_rows,
        total_tasks=len(canonical),
        title=line(view.options.title or f"SmartSOM · {view.name}"),
        subtitle=line(
            getattr(view, "observation_notice", None)
            or ("控制模式" if view.controlling else "只读监控")
            + " · "
            + runtime_status(state.get("status", view.status))
        ),
        reason=line("调度原因 · " + str(reason)),
        feedback=[
            line(value, feedback_style) for value in runtime_feedback_lines(view)
        ],
    )


def render(view):
    if (view.tuning or {}).get("runtime_control"):
        return runtime_render(view)
    summary = view.tuning or {}
    from smartsom.telemetry.execution_metrics import batch_text

    stage = summary.get("stage", view.stage)
    calibration = summary.get("calibration", {})
    resources = summary.get("resources", {})
    measured, candidates = candidate_counts(calibration)
    pairs = _entry_rows(view)
    selected = view.task_pages.ordered(pairs)
    finished = sum(pair[1]["status"] in FINAL for pair in pairs)
    successful = sum(pair[1]["status"] == "completed" for pair in pairs)
    online = online_mode(summary)
    calibrating = has_calibration_stage(view) and stage in {
        "calibration",
        "calibrating",
    }
    waiting = has_calibration_stage(view) and stage in WAITING
    condensed = view.console.height <= 24
    tiny = view.console.height <= 12
    profile = profile_summary(calibration.get("recommendation"))
    if calibrating or waiting:
        stage_items = [
            line(
                f"性能档 {shown(calibration.get('level', 'quick'))} · "
                f"混组调度 {shown(calibration.get('schedule_status', '未判定'))}",
                "#f0a940",
            ),
            line(
                f"耗时/预算 {seconds_text(calibration.get('wall_seconds'))}/{seconds_text(calibration.get('limit_seconds'))}s"
                f" · 已测候选 {shown(measured)}/{shown(candidates)}"
                f" · 等待资源 {duration(calibration.get('waiting_seconds'))}",
                "#f0a940",
            ),
        ]
        if not tiny:
            stage_items.append(
                bar(
                    "候选测量",
                    measured,
                    candidates,
                    color="#f0a940",
                )
            )
            stage_items.append(line(f"待运行实验 {len(pairs)} · 已结束 {finished}"))
        if not condensed:
            stage_items.extend(
                [
                    line(
                        f"当前 {shown(calibration.get('phase'))} · 剩余 {duration(calibration.get('remaining_seconds'))}"
                        f" · 候选 {profile_summary(calibration.get('profile'))}",
                        "#f0a940",
                    ),
                    line(resource_text(resources), "dim"),
                ]
            )
        stage_title = "当前阶段 · 性能评估" if calibrating else "当前阶段 · 等待资源"
        stage_color = "#f0a940"
    elif view.status == "recommended":
        stage_items = [line("仅生成资源建议；未启动正式实验")]
        if not tiny:
            stage_items.append(line("推荐设置 " + profile, "cyan"))
        stage_title = "阶段结果 · 性能评估"
        stage_color = "#f0a940"
    else:
        stage_items = [
            line(f"正式实验结束 {finished}/{len(pairs)} · 成功 {successful}"),
        ]
        if online and condensed and not tiny:
            stage_items = []
        if not condensed and not online and not summary.get("manual"):
            stage_items.append(
                line(
                    f"性能档 {shown(calibration.get('level', 'quick'))} · "
                    f"混组调度 {shown(calibration.get('schedule_status', '未判定'))}",
                    "cyan",
                )
            )
        if not tiny:
            stage_items.append(line(timeline_detail(view, "formal"), "#70b7ee"))
            if not (online and condensed):
                stage_items.append(
                    line(
                        actual_allocation(pairs, profile_mode=resources.get("mode")),
                        "cyan",
                    )
                )
        if (
            not tiny
            and not online
            and not summary.get("manual")
            and not calibration.get("calibrated", True)
        ):
            stage_items.append(line("起始设置（未经校准） " + profile, "yellow"))
        if summary.get("manual"):
            stage_items.extend(
                line(value, "cyan")
                for value in manual_lines(summary, condensed=condensed)
            )
            if view.controlling:
                stage_items.append(
                    line("c 修改总并发 · 输入数字后 Enter 确认 · d 离开界面", "dim")
                )
        if online:
            if tiny:
                stage_items = [
                    line("性能 · 在线调整", "cyan"),
                    line("等待真实更新与内存观测", "cyan"),
                ]
                groups = _online_groups(summary)
                if groups:
                    group = groups[0][1]
                    rate = group.get("latest_throughput")
                    stage_items = [
                        line(
                            f"性能 · 在线调整 · 运行/上限/封顶 {shown(group.get('active'))}/{shown(group.get('limit'))}/{shown(group.get('ceiling'))}",
                            "cyan",
                        ),
                        line(
                            f"最近窗口 {metric_text(rate)} ticks/s · {shown(group.get('measured_concurrency'))} 个实验"
                            if rate is not None
                            else "最近完整窗口：待测",
                            "cyan",
                        ),
                    ]
            else:
                stage_items.append(performance_card(summary, condensed=condensed))
        stage_title = (
            "当前阶段 · 训练与最终评估"
            if stage not in FINAL
            else "阶段结果 · 训练与最终评估"
        )
        stage_color = "#70b7ee"
    pending = sum(bool(entry.get("pending_resize")) for entry, _ in pairs)
    notes = []
    if pending and not tiny and not (online and condensed):
        notes.append(f"资源调整待生效 {pending} 个请求；实际分配尚未改变")
    if online and stage in WAITING and not condensed:
        notes.append("等待资源；继续使用完整更新观测吞吐")
    if calibration.get("reason") and not condensed and (calibrating or waiting):
        notes.append("校准说明 " + compact(calibration["reason"]))
    apps = suggestions(summary)
    if apps and not condensed:
        notes.append(
            "可手动关闭非必要应用后重新检测："
            + "；".join(
                f"{app['name']} PID {shown(app.get('pid'))} ({shown(app.get('cpu_cores'))} cores)"
                for app in apps
            )
        )
    # Reserve one diagnostic row. Missing/healthy samples must not move cards.
    if not tiny and (not condensed or calibrating or waiting):
        stage_items.append(line(" · ".join(notes), "yellow"))
    if summary.get("runtime_control"):
        stage_items.extend(
            line(value, "cyan")
            for value in runtime_lines(
                summary, controlling=view.controlling, condensed=tiny
            )
        )
    base = [
        render_timeline(view),
        Panel(
            Group(*stage_items),
            title=line(stage_title, "bold"),
            border_style=stage_color,
            padding=(0, 1),
        ),
    ]
    show_trials = not (calibrating or waiting or view.status == "recommended")
    cells, compact_cells = [], []
    if show_trials:
        for entry, row in selected:
            if condensed:
                cells.append(compact_trial_card(view, entry, row))
            else:
                content = (
                    card(view, row, reserve_evaluation=True)
                    if row["id"] in view.tasks
                    else Panel(
                        line("尚未启动 · 进度 N/A · ETA 待启动"),
                        title=line(row["name"]),
                        height=9,
                    )
                )
                detail = [content, line(allocation_text(entry), "dim")]
                detail.append(
                    line(
                        entry.get("resource_change_reason") or ""
                        if entry.get("status") not in FINAL
                        else "",
                        "yellow",
                    )
                )
                cells.append(Group(*detail))
            compact_cells.append(
                Panel(
                    Group(
                        line(f"{row['id']} · {row['status']}"),
                        line(
                            f"{shown(row.get('completed'))}/{shown(row.get('total'))} {row.get('unit', '')}"
                        ),
                        line(row.get("reason") or "", "red"),
                    ),
                    padding=(0, 1),
                    border_style="#74aaff",
                )
            )
    base.append(line(batch_text(view), "cyan"))
    short = [
        line(
            f"{STAGES.get(stage, stage)} · 正式实验结束 {finished}/{len(pairs)} · 成功 {successful}",
            "bold",
        )
    ]
    if summary.get("runtime_control"):
        short.extend(
            line(value, "cyan")
            for value in runtime_lines(
                summary, controlling=view.controlling, condensed=True
            )
        )
    elif summary.get("manual"):
        short += [
            line(
                f"调度 · 手动上限 · 实际运行 {shown(summary.get('active'))}/{shown(summary.get('concurrency_limit'))}",
                "cyan",
            ),
            line(
                f"CPU {metric_text(resources.get('cpus_used'))}/{metric_text(resources.get('cpus_total'))} 核 · 总吞吐 {metric_text(summary.get('total_ticks_per_second'))} ticks/s",
                "cyan",
            ),
        ]
    elif online:
        short.append(line("性能 · 在线调整", "cyan"))
        short.extend(
            line(row, "cyan") for row in performance_lines(summary, condensed=True)[:2]
        )
    short.append(line(batch_text(view), "cyan"))
    return render_panel(
        view,
        header=base,
        ids=[row["id"] for _, row in selected] if show_trials else [],
        cards=cells,
        compact_cards=compact_cells,
        short_header=short,
        title=line(view.options.title or f"SmartSOM · {view.name}"),
        subtitle=line(f"{STAGES.get(stage, stage)} [{view.status}]"),
        footer=control_footer(view, summary)
        + "d 离开界面 · Ctrl+C×2 "
        + ("安全停止任务" if view.controlling else "关闭监控"),
    )
