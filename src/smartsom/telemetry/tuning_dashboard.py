"""Read-only presentation of calibration and trial resource facts; no Ray imports."""

import json
import math
from copy import deepcopy

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

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
from smartsom.telemetry.timeline import detail as timeline_detail
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
}
PROCESS_FIELDS = {"pid", "name", "cpu_cores", "rss", "protected"}
COUNT_FIELDS = {
    "active_seconds",
    "wall_seconds",
    "remaining_seconds",
    "limit_seconds",
    "waiting_seconds",
    "cpus_available",
    "memory_available",
    "external_cpu_load",
    "threads",
    "requested_cpus",
    "actual_cpus",
    "allocation_epoch",
    "pid",
    "cpu_cores",
    "rss",
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
        return "尚未分配正式实验资源"
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


def summary_lines(summary):
    calibration = summary.get("calibration", {})
    measured, candidates = candidate_counts(calibration)
    rows = [
        f"调度阶段 {STAGES.get(summary.get('stage'), summary.get('stage', 'N/A'))}",
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
    if calibration.get("reason"):
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


def _entry_rows(view):
    entries = (view.tuning or {}).get("entries", [])
    result = []
    for entry in entries:
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
        return Panel(line("尚未启动 · 进度 N/A · ETA 待启动"), title=line(row["name"]))
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
    items = [progress, phase_bar(view, row), line(detail, "dim"), line(resource, "dim")]
    if row.get("reason"):
        items.append(line(row["reason"], "bold red"))
    if entry.get("resource_change_reason") and entry.get("status") not in FINAL:
        items.append(line(entry["resource_change_reason"], "yellow"))
    return Panel(
        Group(*items),
        title=line(
            f"{row['name']} · {PHASE_NAMES.get(row.get('stage'), row.get('stage', 'N/A'))}",
            "bold",
        ),
        box=box.SQUARE,
        border_style="#74aaff",
        padding=(0, 1),
    )


def render(view):
    summary = view.tuning or {}
    stage = summary.get("stage", view.stage)
    calibration = summary.get("calibration", {})
    resources = summary.get("resources", {})
    measured, candidates = candidate_counts(calibration)
    pairs = _entry_rows(view)
    active = [pair for pair in pairs if pair[1]["status"] not in FINAL]
    ended = [pair for pair in pairs if pair[1]["status"] in FINAL]
    selected = active or list(reversed(ended))[:3]
    finished = sum(pair[1]["status"] in FINAL for pair in pairs)
    successful = sum(pair[1]["status"] == "completed" for pair in pairs)
    calibrating = stage in {"calibration", "calibrating"}
    waiting = stage in WAITING
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
        if not condensed:
            stage_items.append(
                line(
                    f"性能档 {shown(calibration.get('level', 'quick'))} · "
                    f"混组调度 {shown(calibration.get('schedule_status', '未判定'))}",
                    "cyan",
                )
            )
        if not tiny:
            stage_items.append(line(timeline_detail(view, "formal"), "#70b7ee"))
            stage_items.append(
                line(
                    actual_allocation(pairs, profile_mode=resources.get("mode")), "cyan"
                )
            )
        if not tiny and not calibration.get("calibrated", True):
            stage_items.append(line("起始设置（未经校准） " + profile, "yellow"))
        stage_title = (
            "当前阶段 · 训练与最终评估"
            if stage not in FINAL
            else "阶段结果 · 训练与最终评估"
        )
        stage_color = "#70b7ee"
    pending = sum(bool(entry.get("pending_resize")) for entry, _ in pairs)
    if pending and not tiny:
        stage_items.append(
            line(f"资源调整待生效 {pending} 个请求；实际分配尚未改变", "yellow")
        )
    if calibration.get("reason") and not condensed and (calibrating or waiting):
        stage_items.append(line("校准说明 " + compact(calibration["reason"]), "dim"))
    apps = suggestions(summary)
    if apps:
        stage_items.append(
            line(
                "可手动关闭非必要应用后重新检测："
                + "；".join(
                    f"{app['name']} PID {shown(app.get('pid'))} ({shown(app.get('cpu_cores'))} cores)"
                    for app in apps
                ),
                "yellow",
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
    wide = view.console.width >= 160 and len(selected) > 1
    count = len(selected) if show_trials and not tiny else 0
    height = max(3, view.console.height - 1)

    def build():
        items = list(base)
        if not tiny and show_trials:
            items.append(line(f"实验卡片 · 显示 {count}/{len(selected)}", "bold"))
        visible = selected[:count]
        cells = []
        for entry, row in visible:
            if condensed:
                cells.append(compact_trial_card(view, entry, row))
            else:
                content = (
                    card(view, row)
                    if row["id"] in view.tasks
                    else Panel(
                        line("尚未启动 · 进度 N/A · ETA 待启动"),
                        title=line(row["name"]),
                    )
                )
                detail = [content, line(allocation_text(entry), "dim")]
                if (
                    entry.get("resource_change_reason")
                    and entry.get("status") not in FINAL
                ):
                    detail.append(line(entry["resource_change_reason"], "yellow"))
                cells.append(Group(*detail))
        if wide and cells:
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            for index in range(0, len(cells), 2):
                grid.add_row(
                    cells[index], cells[index + 1] if index + 1 < len(cells) else ""
                )
            items.append(grid)
        else:
            items.extend(cells)
        hidden = len(selected) - count if show_trials else 0
        if hidden:
            items.append(
                line(f"+{hidden} hidden trials · 全部实例见 runtime.log", "yellow")
            )
        if view.notice and not condensed:
            items.append(line(view.notice, "dim"))
        if not tiny:
            items.append(
                line(
                    "d 离开界面 · Ctrl+C×2 "
                    + ("安全停止任务" if view.controlling else "关闭监控"),
                    "dim",
                )
            )
        return Panel(
            Group(*items),
            title=line(view.options.title or f"SmartSOM · {view.name}"),
            subtitle=line(f"{STAGES.get(stage, stage)} [{view.status}]"),
            box=box.SQUARE,
            padding=(0, 1),
        )

    content = build()
    while (
        len(
            view.console.render_lines(
                content, view.console.options.update(height=None), pad=False
            )
        )
        > height
    ):
        if count > 0:
            count -= 1
        else:
            break
        content = build()
    content.height = height
    return content
