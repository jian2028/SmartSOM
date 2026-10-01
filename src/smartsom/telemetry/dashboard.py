"""Shared Rich terminal layout for live execution and read-only monitoring."""

import time

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Column, Table
from rich.text import Text

from smartsom.telemetry.study_progress import duration
from smartsom.telemetry.workflow import task_title

CYAN = "#5fd7dc"
BLUE = "#74aaff"
YELLOW = "#efca72"
AMBER = "#f0a940"
PHASE_NAMES = {
    "sampling": "采样",
    "optimizing": "更新",
    "validation": "验证",
    "evaluation": "评估",
    "saving": "保存",
    "simulation": "运行",
    "initializing": "准备",
    "starting": "启动",
}


def line(value, style=""):
    return Text(str(value), style=style, no_wrap=True, overflow="ellipsis")


def bar(label, completed, total, detail="", *, short=False, overall=False, color=None):
    known = completed is not None and total is not None and total > 0
    fraction = min(1, max(0, completed / total)) if known else None
    columns = [
        TextColumn("{task.description}", markup=False),
        BarColumn(
            bar_width=16 if short else None,
            complete_style=color or (YELLOW if short else CYAN if overall else BLUE),
            finished_style=color or CYAN,
            style="grey37",
        ),
        TextColumn("{task.fields[percent]}", markup=False),
    ]
    if detail:
        columns.append(
            TextColumn(
                "{task.fields[detail]}",
                markup=False,
                table_column=Column(no_wrap=True, overflow="ellipsis", ratio=1),
            )
        )
    progress = Progress(*columns, expand=not short)
    progress.add_task(
        label,
        total=total if known else 1,
        completed=min(completed, total) if known else 0,
        percent=f"{fraction:.0%}"
        if short and known
        else f"{fraction:.1%}"
        if known
        else "N/A",
        detail=detail,
    )
    return progress


def facts(view, row):
    values = row.get("values", {})
    plan = values.get("workflow") or view.workflow or {}
    case = values.get("study_case", {})
    if case:
        plan = {
            **plan,
            **(view.workflow or {}),
            "mode": "study-entry",
            "training_seed": plan.get("training_seed"),
        }
    if view.kind == "run":
        plan = {**plan, "mode": "run", "training_total": 0}
    algorithm = (
        case.get("algorithm") or values.get("algorithm") or plan.get("algorithm", "N/A")
    )
    seed = case.get("seed")
    if seed is None:
        seed = plan.get("training_seed")
        if seed is None:
            seed = (view.workflow or {}).get("training_seed", "N/A")
    done = values.get("updates", values.get("ppo_updates"))
    completed = row.get("completed")
    size, total = plan.get("round_size", 0), plan.get("round_total", 0)
    phase = row.get("stage", "initializing")
    phase = {
        "updating": "optimizing",
        "learning_metrics": "optimizing",
        "evaluating": "evaluation",
    }.get(phase, phase)
    if view.kind == "run" and phase == "evaluation":
        phase = "simulation"
    fields = {
        "id": row["id"],
        "name": row["name"],
        "algorithm": algorithm,
        "seed": seed,
        "H": case.get("H", "N/A"),
        "V": case.get("V", "N/A"),
        "travel": case.get("travel", plan.get("travel", "N/A")),
    }
    default = (
        f"{algorithm} · H={fields['H']} V={fields['V']} · {fields['travel']} · Seed {seed}"
        if case
        else f"{row['name']} · {algorithm} · Seed {seed}"
    )
    template = (
        view.options.task_title
        or plan.get("task_title")
        or (view.workflow or {}).get("task_title")
    )
    name = task_title(template, fields, default)
    return values, plan, phase, algorithm, done, completed, size, total, name


def stage_line(view, row):
    values, plan, phase, algorithm, _, _, _, _, _ = facts(view, row)
    text = Text(no_wrap=True, overflow="ellipsis")
    learning = plan.get("mode") != "evaluation" and plan.get("training_total", 0) > 0
    if learning:
        for name, key in (("采样", "sampling"), ("更新", "optimizing")):
            if key == "optimizing":
                text.append(" ⇄ ", style=BLUE)
            active = phase == key or (
                phase == "sampling"
                and algorithm.upper() == "DQN"
                and key == "optimizing"
                and values.get("optimization_steps", 0) > 0
            )
            text.append(name, style=f"bold {YELLOW}" if active else "dim")
        text.append(" │ ", style="dim")
        batches = values.get("validation_batches_finished")
        text.append(
            f"验证 {batches if batches is not None else 'N/A'}/{plan.get('validation_rounds', 'N/A')}批",
            style=f"bold {YELLOW}" if phase == "validation" else "dim",
        )
        text.append(" │ ", style="dim")
    if plan.get("mode") == "run":
        text.append(
            "运行 → ", style=f"bold {YELLOW}" if phase == "simulation" else "dim"
        )
    text.append("保存", style=f"bold {YELLOW}" if phase == "saving" else "dim")
    if plan.get("mode") in {"study", "study-entry", "evaluation", "train-evaluate"}:
        text.append(" │ ", style="dim")
        text.append(
            f"评估 {values.get('evaluation_finished', 0)}/{values.get('evaluation_requested', plan.get('evaluation_cases', 'N/A'))}",
            style=f"bold {YELLOW}" if phase == "evaluation" else "dim",
        )
    return text


def cycle_line(view, row):
    _, plan, phase, _, done, _, _, total, _ = facts(view, row)
    if not plan.get("training_total") or plan.get("mode") == "evaluation":
        return line(f"当前阶段：{PHASE_NAMES.get(phase, phase)}", "bold")
    value = f"循环已完成 {done if done is not None else 'N/A'}/{total or 'N/A'}轮"
    if (
        done is not None
        and total
        and done < total
        and phase in {"sampling", "optimizing"}
    ):
        value += f" · 当前第{done + 1}轮"
    return line(value, "bold")


def next_line(view, row):
    values, plan, phase, _, done, _, _, total, _ = facts(view, row)
    every = plan.get("validation_every", 0)
    if phase == "validation":
        after = (
            "之后返回训练"
            if done is not None and done < total
            else "之后保存 → 最终评估"
            if plan.get("mode") in {"study-entry", "train-evaluate"}
            else "之后保存 → 结束"
        )
        return line(
            f"正在第{values.get('validation_round', 'N/A')}/{plan.get('validation_rounds', 'N/A')}批验证 · 每批{plan.get('validation_cases', 'N/A')}案例 · {after}",
            "dim",
        )
    if (
        phase == "saving"
        and every
        and done is not None
        and done > 0
        and done % every == 0
        and (values.get("validation_batches_finished", 0) or 0)
        < done // every
        <= plan.get("validation_rounds", 0)
    ):
        return line(
            f"本组循环已完成 → 第{done // every}批验证（{plan.get('validation_cases', 'N/A')}案例）",
            "dim",
        )
    if phase == "evaluation":
        return line(
            f"{values.get('evaluation_kind', '最终评估')} · {values.get('evaluation_requested', plan.get('evaluation_cases', 'N/A'))}案例",
            "dim",
        )
    if every and done is not None and total and done < total:
        remaining = min(every - done % every, total - done)
        if done // every >= plan.get("validation_rounds", 0):
            return line(
                f"距训练预算结束剩{total - done}轮（含当前）；不再有计划验证", "dim"
            )
        return line(
            f"本组第{done % every + 1}/{every}轮 · 距第{done // every + 1}批验证剩{remaining}轮（含当前）",
            "dim",
        )
    return line(f"{row.get('status', 'running')} · {phase}", "dim")


def phase_bar(view, row):
    values, plan, phase, _, done, completed, size, _, _ = facts(view, row)
    if phase == "simulation":
        tick = values.get("evaluation_tick", completed)
        return bar(
            "当前运行",
            tick,
            plan.get("run_total"),
            f"{tick if tick is not None else 'N/A'} ticks",
            short=True,
        )
    if phase in {"validation", "evaluation"}:
        ended, requested = (
            values.get(phase + "_finished", 0),
            values.get(phase + "_requested"),
        )
        tick, limit = values.get(phase + "_tick", 0), values.get(phase + "_tick_limit")
        fraction = (
            min(1, tick / limit)
            if limit
            and values.get(phase + "_case_active", True)
            and ended < (requested or 0)
            else 0
        )
        detail = f"结束{ended}/{requested or 'N/A'}"
        if limit:
            detail += f" · 本案例{tick:,}/{limit:,} ticks"
        return bar(
            "当前验证" if phase == "validation" else "当前评估",
            ended + fraction,
            requested,
            detail,
            short=True,
        )
    if phase == "sampling" and size and done is not None and completed is not None:
        local = min(size, max(0, completed - done * size))
        # The last collection interval may be shorter than round_size.
        limit = min(size, max(1, plan.get("training_total", size) - done * size))
        return bar(
            "当前采样",
            local,
            limit,
            f"本轮{local:,}/{limit:,} {plan.get('training_unit', 'steps')}",
            short=True,
        )
    from smartsom.telemetry.runtime import FINAL

    finished = row.get("status") in FINAL
    endpoint = row.get("updated_at", time.time()) if finished else time.time()
    seconds = max(
        0, int(endpoint - row.get("stage_started_at", row.get("updated_at", endpoint)))
    )
    return line(
        (
            f"{PHASE_NAMES.get(phase, phase)} [{row['status']}] · 阶段耗时{duration(seconds)}"
            if finished
            else f"当前{PHASE_NAMES.get(phase, phase)} · 已用{duration(seconds)}"
        ),
        YELLOW,
    )


def identity(view, row):
    return facts(view, row)[-1]


def task_eta(view, row):
    """Estimate a task's remaining wall time from its own completed work."""
    from smartsom.telemetry.runtime import FINAL

    if row.get("status") in FINAL:
        return "已完成" if row.get("status") == "completed" else "—"
    if row.get("status") in {"queued", "pending"}:
        return "待启动"
    started = row.get("started_at")
    elapsed = max(0, time.time() - started) if isinstance(started, (int, float)) else 0
    if elapsed < 20:
        return "估算中"
    values = row.get("values", {})
    planned_done, planned_total = (
        values.get("planned_work_completed"),
        values.get("planned_work_total"),
    )
    if (
        isinstance(planned_total, (int, float))
        and planned_total > 0
        and isinstance(planned_done, (int, float))
    ):
        fraction = min(1.0, max(0.0, planned_done / planned_total))
    else:
        plan = values.get("workflow") or view.workflow or {}
        mode = plan.get("mode", "train")
        parts = []
        if mode not in {"evaluation", "run"}:
            total = row.get("total") or plan.get("training_total")
            done = row.get("completed")
            parts.append(min(1.0, done / total) if total and done is not None else 0.0)
        if mode in {"evaluation", "train-evaluate", "study-entry"}:
            requested = values.get("evaluation_requested") or plan.get(
                "evaluation_cases"
            )
            done = values.get("evaluation_finished", 0)
            parts.append(min(1.0, done / requested) if requested else 0.0)
        if mode == "run":
            total = row.get("total") or plan.get("run_total")
            done = values.get("evaluation_tick", row.get("completed"))
            parts.append(min(1.0, done / total) if total and done is not None else 0.0)
        fraction = sum(parts) / len(parts) if parts else 0.0
    if fraction < 0.01 or fraction >= 1:
        return "估算中"
    return "≈ " + duration(min(30 * 86400, elapsed * (1 - fraction) / fraction))


def running_allocation(rows, metadata):
    """Summarize the layout reported by workers, or label the planned layout."""
    reported = [
        row.get("values", {}).get("runtime_mode")
        for row in rows
        if row.get("status") == "running"
        and isinstance(row.get("values", {}).get("runtime_mode"), str)
    ]
    if not reported:
        planned = metadata.get("runtime_mode")
        return "资源配置待上报" + (f" · 预设 {planned}" if planned else "")
    if len(set(reported)) != 1:
        return "实际资源：各实验配置不同（见实例记录）"
    fields = {}
    for part in reported[0].replace(" · ", ";").split(";"):
        key, separator, value = part.strip().partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    device = {"cuda": "GPU/CUDA", "mps": "GPU/MPS"}.get(
        fields.get("device"), fields.get("device", "未知设备").upper()
    )
    return (
        f"实际运行 {device}"
        f" · 并行实验 {len(reported)}"
        f" · 每实验环境 {fields.get('num_envs', fields.get('envs', '?'))}"
        f" · 采样进程 {fields.get('sampling_processes', '?')}"
        f" · 计算线程 {fields.get('numerical_threads', fields.get('threads', '?'))}"
    )


def card(view, row, *, solo=False):
    values, plan, _, _, _, completed, _, _, name = facts(view, row)
    budget = row.get("total")
    if view.kind == "run":
        completed = values.get("evaluation_tick", completed)
        budget = plan.get("run_total")
    entry_work = values.get("planned_work_completed")
    entry_total = values.get("planned_work_total")
    label = (
        "实例累计"
        if entry_total
        else "评估案例"
        if plan.get("mode") == "evaluation"
        else "运行预算"
        if view.kind == "run"
        else "训练预算"
    )
    progress = bar(
        label, entry_work if entry_total else completed, entry_total or budget
    )
    items = [
        progress,
        cycle_line(view, row),
        stage_line(view, row),
        next_line(view, row),
        phase_bar(view, row),
    ]
    unit = (
        "physical ticks"
        if view.kind == "run"
        else row.get("unit", plan.get("training_unit", "progress"))
    )
    counters = f"{unit}: {completed if completed is not None else 'N/A'}/{budget if budget is not None else 'N/A'}"
    counters += f" · ETA {task_eta(view, row)}"
    if plan.get("mode") != "evaluation" and view.kind != "run":
        if "optimization_steps" in values:
            counters += f" · 优化步数 {values['optimization_steps']}"
        elif "learner_updates" in values:
            counters += f" · 学习器更新 {values['learner_updates']}"
        else:
            counters += " · 优化步数 N/A"
    items.append(line(counters, "dim"))
    if row.get("reason"):
        items[-1] = line(row["reason"], "bold red")
    if solo and any(
        k in values
        for k in ("training_deliveries", "episode_return", "evaluation_completed")
    ):
        items.append(
            line(
                f"交付 {values.get('training_deliveries', 'N/A')} · 最近回报 {values.get('episode_return', 'N/A')} · 评估成功 {values.get('evaluation_completed', 'N/A')}",
                "dim",
            )
        )
    return Panel(
        Group(*items),
        title=line(name, "bold"),
        box=box.SQUARE,
        border_style=BLUE,
        padding=(0, 1),
        height=9 if len(items) > 6 else 8,
    )


def table(view, rows):
    narrow = view.console.width < 100
    result = Table(box=None, expand=True, padding=(0, 1), header_style=f"bold {CYAN}")
    result.add_column("实例", ratio=2, no_wrap=True, overflow="ellipsis")
    result.add_column("Seed", width=5, no_wrap=True, overflow="ellipsis")
    result.add_column("轮次", width=7, no_wrap=True)
    result.add_column("阶段", width=8, no_wrap=True, overflow="ellipsis")
    if not narrow:
        result.add_column("距验证", width=9, no_wrap=True)
    result.add_column("阶段进度", width=13, no_wrap=True, overflow="ellipsis")
    for row in rows:
        values, plan, phase, _, done, completed, size, total, name = facts(view, row)
        names = {
            "sampling": "采样",
            "optimizing": "更新",
            "validation": "验证",
            "evaluation": "评估",
            "saving": "保存",
        }
        phase_text = names.get(phase, phase)
        if phase in {"validation", "evaluation"}:
            ended, requested = (
                values.get(phase + "_finished", 0),
                values.get(phase + "_requested"),
            )
            detail = f"{ended}/{requested if requested is not None else 'N/A'}"
        elif (
            phase == "sampling" and size and done is not None and completed is not None
        ):
            detail = f"{min(size, max(0, completed - done * size)) / min(size, max(1, plan.get('training_total', size) - done * size)):.0%}"
        else:
            detail = duration(
                max(
                    0,
                    time.time()
                    - row.get("stage_started_at", row.get("updated_at", time.time())),
                )
            )
        seed = values.get("study_case", {}).get("seed")
        if seed is None:
            seed = plan.get("training_seed", "N/A")
        cells = [
            line(name),
            line(seed),
            line(f"{done if done is not None else 'N/A'}/{total or 'N/A'}"),
            line(phase_text, YELLOW),
        ]
        if not narrow:
            every = plan.get("validation_every", 0)
            remaining = (
                f"{every - done % every}轮"
                if every and done is not None and total and done < total
                else "—"
            )
            cells.append(line(remaining))
        cells.append(line(detail))
        result.add_row(*cells)
    return result


def render(view):
    from smartsom.telemetry.runtime import FINAL

    width, height = view.console.width, max(3, view.console.height - 1)
    linked = view.preflight is not None or view.tuning is not None
    metadata = view.workflow or {}
    rows = list(view.tasks.values())
    if view.kind in {"evaluation", "train-evaluate"} and any(
        r["id"] == "evaluation" for r in rows
    ):
        rows = [
            r
            for r in rows
            if r["id"] == "evaluation"
            or r.get("unit")
            in {"environment steps", "adapter decisions", "sampling decisions"}
            or r["id"] == "training"
        ]
    active = [r for r in rows if r["status"] not in FINAL | {"pending", "queued"}]
    ended = [r for r in rows if r["status"] in FINAL]
    selected = active or list(reversed(ended))[:3]
    queued = sum(r["status"] in {"pending", "queued"} for r in rows) + max(
        0, (view.total_tasks or len(rows)) - len(rows)
    )
    overview = view.overview or {}
    factories = (
        " · ".join(filter(None, (metadata.get("factory"), metadata.get("map"))))
        or "pending"
    )
    algorithms = "/".join(metadata.get("algorithms", [])) or metadata.get(
        "algorithm", "N/A"
    )
    common = f"Factory {factories} │ {algorithms}"
    if metadata.get("H_cases"):
        common += f" │ H {'/'.join(metadata['H_cases'])} V {'/'.join(metadata.get('V_cases', []))}"
    if metadata.get("social_information"):
        common += " │ Social " + metadata["social_information"]
    seed_info = f"Train seeds {metadata.get('training_seeds', [])} │ Val root {metadata.get('validation_seed', 'N/A')}: {metadata.get('validation_cases', 0)}/批 │ Eval root {metadata.get('evaluation_seed', 'N/A')}: {metadata.get('evaluation_cases', 0)}"
    if metadata.get("mode") == "run":
        seed_info = f"Run seed {metadata.get('training_seed', 'N/A')} │ {metadata.get('run_total', 'N/A')} physical ticks"
    if metadata.get("mode") == "evaluation":
        seed_info = f"Evaluation root {metadata.get('evaluation_seed', 'N/A')} │ {metadata.get('evaluation_cases', 'N/A')}案例"
    base = [
        line(common, "dim"),
        line(seed_info, CYAN),
        bar(
            "总流程",
            overview.get("work_completed"),
            overview.get("work_total"),
            overall=True,
        ),
        line(
            f"Elapsed {duration(overview.get('elapsed_seconds'))} │ ETA ≈ {duration(overview.get('eta_seconds'))}",
            f"bold {CYAN}",
        ),
    ]
    if linked:
        from smartsom.telemetry.timeline import render as render_timeline

        base.insert(0, render_timeline(view))
        base.append(
            line(
                "d 离开界面 · Ctrl+C×2 "
                + ("安全停止任务" if view.controlling else "关闭监控"),
                "dim",
            )
        )
    if metadata.get("runtime_mode"):
        base.insert(2, line(metadata["runtime_mode"], "dim"))
    finished, successful = view.task_counts()
    if view.kind == "study" or view.total_tasks is not None:
        base.append(
            line(
                f"完成 {finished}/{view.total_tasks or len(rows)} · 成功 {successful} · 运行 {len(active)}/{metadata.get('max_concurrent', len(active))} · 排队 {queued} · 失败 {sum(r['status'] == 'failed' for r in ended)}"
            )
        )
    else:
        base.append(line(f"{view.stage} [{view.status}]"))
    if metadata.get("training_total") and metadata.get("mode") not in {
        "evaluation",
        "run",
    }:
        cadence = f"采样 ⇄ 更新 ↻ {metadata.get('round_total', 'N/A')}轮 · {metadata.get('round_size', 'N/A')} {metadata.get('training_unit', 'steps')}/轮"
        if metadata.get("validation_rounds"):
            cadence += f" · 每{metadata.get('validation_every')}轮验证{metadata.get('validation_cases')}案例"
        base.append(line(cadence, "dim"))
    if linked:
        from smartsom.telemetry.timeline import detail as timeline_detail
        from smartsom.telemetry.timeline import render as render_timeline

        phase_items = [line(timeline_detail(view, "formal"), BLUE)]
        if height > 24:
            phase_items.extend([line(common, "dim"), line(seed_info, CYAN)])
        if height > 12:
            phase_items.append(line(running_allocation(rows, metadata), "cyan"))
        if height > 12:
            phase_items.append(
                line(
                    f"完成 {finished}/{view.total_tasks or len(rows)} · 成功 {successful} · 排队 {queued}"
                )
            )
        base = [
            render_timeline(view),
            Panel(
                Group(*phase_items),
                title=line(
                    "阶段结果 · 训练与最终评估"
                    if view.status in FINAL
                    else "当前阶段 · 训练与最终评估",
                    "bold",
                ),
                border_style=BLUE,
                padding=(0, 1),
            ),
        ]
    wide = width >= 180 and len(selected) > 1
    count = 0 if linked and height <= 12 else len(selected)
    focused = bool(selected) and width < 180

    def build(n, focus):
        items = list(base)
        visible = selected[:n]
        if linked and height > 12:
            items.append(line(f"实验卡片 · 显示 {n}/{len(selected)}", "bold"))
        if linked and not visible:
            pass
        elif wide:
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=1)
            grid.add_column(ratio=1)
            for index in range(0, len(visible), 2):
                grid.add_row(
                    card(view, visible[index]),
                    card(view, visible[index + 1]) if index + 1 < len(visible) else "",
                )
            items.append(grid)
        elif linked and visible:
            items.extend(card(view, row, solo=len(visible) == 1) for row in visible)
        elif len(visible) == 1 and width >= 100:
            items.append(card(view, visible[0], solo=True))
        else:
            items.append(table(view, visible))
            if focus and visible:
                items.extend(
                    [
                        line("详情：" + identity(view, visible[0]), "dim"),
                        phase_bar(view, visible[0]),
                    ]
                )
        omitted = len(selected) - n
        if omitted:
            items.append(
                line(f"+{omitted} other active tasks · 全部实例见 runtime.log", YELLOW)
            )
        if view.notice:
            items.append(line(view.notice, "dim"))
        if linked and height > 12:
            items.append(
                line(
                    "d 离开界面 · Ctrl+C×2 "
                    + ("安全停止任务" if view.controlling else "关闭监控"),
                    "dim",
                )
            )
        else:
            items.append(line("阶段栏高亮当前状态 · 检查点与详细指标见日志", "dim"))
        return Panel(
            Group(*items),
            title=line(
                view.options.title or metadata.get("title") or f"SmartSOM · {view.name}"
            ),
            subtitle=line(f"{view.stage} [{view.status}]"),
            box=box.SQUARE,
            padding=(0, 1),
        )

    content = build(count, focused)
    while (
        len(
            view.console.render_lines(
                content, view.console.options.update(height=None), pad=False
            )
        )
        > height
    ):
        if focused:
            focused = False
        elif count > 0:
            count -= 1
        elif len(base) > 3:
            base.pop(-1)
        else:
            break
        content = build(count, focused)
    content.height = height
    return content
