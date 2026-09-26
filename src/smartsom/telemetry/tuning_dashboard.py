"""Read-only presentation of calibration and trial resource facts; no Ray imports."""

import json
import math
from copy import deepcopy

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from smartsom.telemetry.dashboard import bar, card, line, phase_bar
from smartsom.telemetry.runtime import FINAL, shown
from smartsom.telemetry.study_progress import duration

STAGES = {
    "preflight": "预检",
    "calibration": "短校准",
    "calibrating": "短校准",
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
        f"可用 CPU {shown(resources.get('cpus_available'))} cores · "
        f"内存 {memory(resources.get('memory_available'))} · "
        f"外部 CPU 负载 {shown(resources.get('external_cpu_load'))} · "
        f"模式 {shown(resources.get('mode'))}"
    )


def allocation_text(entry):
    value = (
        f"线程 {shown(entry.get('threads'))} · "
        f"CPU 请求 {shown(entry.get('requested_cpus'))} / 实际 {shown(entry.get('actual_cpus'))} · "
        f"分配版本 {shown(entry.get('allocation_epoch'))}"
    )
    if entry.get("pending_resize"):
        value += " · 已请求，待生效"
    return value


def summary_lines(summary):
    calibration = summary.get("calibration", {})
    measured, candidates = candidate_counts(calibration)
    rows = [
        f"调度阶段 {STAGES.get(summary.get('stage'), summary.get('stage', 'N/A'))}",
        f"校准活动 {shown(calibration.get('active_seconds'))}/{shown(calibration.get('limit_seconds'))}s；等待资源 {shown(calibration.get('waiting_seconds'))}s；已测候选 {shown(measured)}/{shown(candidates)}",
        "推荐设置 " + compact(calibration.get("recommendation")),
        resource_text(summary.get("resources", {})),
    ]
    if calibration.get("reason"):
        rows.append("校准说明 " + compact(calibration["reason"]))
    for entry in summary.get("entries", []):
        rows.append(
            f"{entry['experiment_id']}: {shown(entry.get('status'))} · {allocation_text(entry)}"
        )
        if entry.get("resource_change_reason"):
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


def trial_table(view, pairs):
    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("实验 / 状态", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("训练预算", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("实际/请求 CPU", ratio=1, no_wrap=True, overflow="ellipsis")
    if view.console.width >= 100:
        table.add_column("验证 / 评估", ratio=2, no_wrap=True, overflow="ellipsis")
    for entry, row in pairs:
        values = row.get("values", {})
        progress = f"{shown(row.get('completed'))}/{shown(row.get('total'))} {row.get('unit', 'N/A')}"
        phase = row.get("stage", "N/A")
        name = f"{row['name']} · {phase} [{row.get('status', 'unknown')}]"
        resource = (
            f"{shown(entry.get('actual_cpus'))}/{shown(entry.get('requested_cpus'))}"
        )
        if entry.get("pending_resize"):
            resource += " 待生效"
        cells = [line(name), line(progress), line(resource)]
        if view.console.width >= 100:
            cases = " · ".join(
                f"{label} {shown(values.get(prefix + '_finished'))}/{shown(values.get(prefix + '_requested'))}"
                for label, prefix in (("验", "validation"), ("评", "evaluation"))
            )
            cells.append(line(cases))
        table.add_row(*cells)
    return table


def candidate_table(calibration, limit):
    rows = calibration.get("candidates")
    if not isinstance(rows, list):
        return None
    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("候选", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("测量 / 状态", ratio=3, no_wrap=True, overflow="ellipsis")
    for index, row in enumerate(rows[:limit]):
        if isinstance(row, dict):
            name = row.get("candidate_id", row.get("id", f"候选 {index + 1}"))
            detail = {
                key: row[key]
                for key in (
                    "status",
                    "threads",
                    "num_envs",
                    "sampling_processes",
                    "throughput",
                    "samples_per_second",
                    "reason",
                )
                if key in row
            }
            table.add_row(line(name), line(compact(detail) if detail else "N/A"))
        else:
            table.add_row(line(row), line("N/A"))
    return table


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
    base = [
        line(
            f"预检 → 短校准 → 正式实验 · 当前 {STAGES.get(stage, stage)} [{view.status}]"
        ),
        line(resource_text(resources), "dim"),
        bar(
            f"校准活动 {shown(calibration.get('active_seconds'))}/{shown(calibration.get('limit_seconds'))}s",
            calibration.get("active_seconds"),
            calibration.get("limit_seconds"),
        ),
        line(f"等待资源 {duration(calibration.get('waiting_seconds'))}", "yellow"),
        bar(f"已测候选 {shown(measured)}/{shown(candidates)}", measured, candidates),
        line("推荐设置 " + compact(calibration.get("recommendation")), "cyan"),
        line(f"正式实验结束 {finished}/{len(pairs)} · 成功 {successful}"),
    ]
    pending = sum(bool(entry.get("pending_resize")) for entry, _ in pairs)
    if pending:
        base.append(
            line(f"资源调整待生效 {pending} 个请求；实际分配尚未改变", "yellow")
        )
    if calibration.get("reason"):
        base.append(line("校准说明 " + compact(calibration["reason"]), "dim"))
    apps = suggestions(summary)
    if apps:
        base.append(
            line(
                "可手动关闭非必要应用后重新检测："
                + "；".join(
                    f"{app['name']} PID {shown(app.get('pid'))} ({shown(app.get('cpu_cores'))} cores)"
                    for app in apps
                ),
                "yellow",
            )
        )
    wide = view.console.width >= 160 and len(selected) > 1
    count = len(selected)
    candidate_limit = 3 if stage in {"calibration", "calibrating"} else 0
    focused = bool(selected)
    height = max(3, view.console.height - 1)

    def build():
        items = list(base)
        candidate_rows = candidate_table(calibration, candidate_limit)
        if candidate_rows is not None and stage in {"calibration", "calibrating"}:
            if candidate_limit:
                items.append(candidate_rows)
            hidden = len(calibration["candidates"]) - candidate_limit
            if hidden > 0:
                items.append(line(f"+{hidden} hidden candidates", "yellow"))
        visible = selected[:count]
        if wide and visible:
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            cells = []
            for entry, row in visible:
                content = (
                    card(view, row)
                    if row["id"] in view.tasks
                    else Panel(line("尚未启动 · 进度 N/A"), title=line(row["name"]))
                )
                detail = [line(allocation_text(entry), "dim"), content]
                if entry.get("resource_change_reason"):
                    detail.insert(1, line(entry["resource_change_reason"], "yellow"))
                cells.append(Group(*detail))
            for index in range(0, len(cells), 2):
                grid.add_row(
                    cells[index], cells[index + 1] if index + 1 < len(cells) else ""
                )
            items.append(grid)
        else:
            items.append(trial_table(view, visible))
            if focused and visible:
                entry, row = visible[0]
                items.append(line(allocation_text(entry), "dim"))
                if row["id"] in view.tasks:
                    items.append(phase_bar(view, row))
                if entry.get("resource_change_reason"):
                    items.append(line(entry["resource_change_reason"], "yellow"))
        hidden = len(selected) - count
        if hidden:
            items.append(
                line(f"+{hidden} hidden trials · 全部实例见 runtime.log", "yellow")
            )
        if view.notice:
            items.append(line(view.notice, "dim"))
        items.append(line("校准结果不是实验完成；资源请求生效前保持实际分配", "dim"))
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
        if candidate_limit:
            candidate_limit -= 1
        elif count > 1:
            count -= 1
        elif focused:
            focused = False
        elif count:
            count -= 1
        elif len(base) > 3:
            base.pop(-1)
        else:
            break
        content = build()
    content.height = height
    return content
