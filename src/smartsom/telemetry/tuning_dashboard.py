"""Read-only presentation of calibration and trial resource facts; no Ray imports."""

import json
import math
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
from smartsom.telemetry.task_pages import render_panel
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
    "cpus_used",
    "owned_cpu_cores",
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


def manual_lines(summary):
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
        f"调度 · 手动上限 · 实际运行 {shown(summary.get('active'))}/{shown(summary.get('concurrency_limit'))} 个任务 · 自适应关闭",
        f"主机可见 CPU 忙碌 {host}/{metric_text(resources.get('cpus_total'))} 核 · 本批次 {owned} 核 · 可用内存 {memory(resources.get('memory_available'))}",
        f"总吞吐 {total} ticks/s · 训练 {training} ticks/s（最近 {metric_text(throughput.get('window_seconds', 30))} 秒）",
        "队列开放 · 命令行 batch-add 追加任务"
        if summary.get("queue_open")
        else "队列已关闭，等待剩余任务结束",
    ]
    if throughput.get("reason"):
        reasons = {
            "warming execution window": "吞吐窗口采集中",
            "source progress is stale": "源进度已过期",
            "source has no execution counters": "源任务尚无执行计数",
            "legacy source has training-only counters": "旧快照仅有训练计数，执行总吞吐待新观测",
        }
        lines.append(reasons.get(throughput["reason"], "执行观测待补齐"))
    if resources.get("owned_cpu_reason"):
        lines.append(
            "批次 CPU "
            + (
                "采集中：新进程首样本"
                if resources["owned_cpu_reason"]
                == "owned process first CPU sample unavailable"
                else "观测不可用：拥有进程记录待核实"
            )
        )
    return lines


def summary_lines(summary):
    calibration = summary.get("calibration", {})
    measured, candidates = candidate_counts(calibration)
    rows = [
        f"调度阶段 {STAGES.get(summary.get('stage'), summary.get('stage', 'N/A'))}",
    ]
    if summary.get("manual"):
        rows.extend(manual_lines(summary))
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
            stage_items.extend(line(value, "cyan") for value in manual_lines(summary))
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
            if stage in WAITING and not condensed:
                stage_items.append(line("等待资源；继续使用完整更新观测吞吐", "yellow"))
        stage_title = (
            "当前阶段 · 训练与最终评估"
            if stage not in FINAL
            else "阶段结果 · 训练与最终评估"
        )
        stage_color = "#70b7ee"
    pending = sum(bool(entry.get("pending_resize")) for entry, _ in pairs)
    if pending and not tiny and not (online and condensed):
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
    cells, compact_cells = [], []
    if show_trials:
        for entry, row in selected:
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
            compact_cells.append(
                Panel(
                    Group(
                        line(f"{row['id']} · {row['status']}"),
                        line(
                            f"{shown(row.get('completed'))}/{shown(row.get('total'))} {row.get('unit', '')}"
                        ),
                        *([line(row["reason"], "red")] if row.get("reason") else []),
                    ),
                    padding=(0, 1),
                    border_style="#74aaff",
                )
            )
    base.append(line(batch_text(view), "cyan"))
    if view.notice:
        base.append(line(view.notice, "dim"))
    short = [
        line(
            f"{STAGES.get(stage, stage)} · 正式实验结束 {finished}/{len(pairs)} · 成功 {successful}",
            "bold",
        )
    ]
    if summary.get("manual"):
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
        footer=("c 修改总并发 · " if summary.get("manual") and view.controlling else "")
        + "d 离开界面 · Ctrl+C×2 "
        + ("安全停止任务" if view.controlling else "关闭监控"),
    )
