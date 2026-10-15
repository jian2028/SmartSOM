"""View-only adaptive Rich grids, stable task anchors and height pagination."""

from dataclasses import dataclass, field

from rich.table import Table

MIN_CARD_WIDTH = 48
GAP = 2


@dataclass
class TaskIdentity:
    """Resolve the selected marker after pagination has chosen its anchor."""

    view: object
    identity: str
    status: str = ""
    failed: bool = False

    def copy(self):
        # Panel resolves its Text title lazily, after page selection.
        return self.__rich__()

    def __rich__(self):
        from rich.text import Text

        selected = self.view.task_pages.anchor == self.identity
        return Text(
            ("▶ " if selected else "  ")
            + (self.status + " · " if self.status else "")
            + self.identity,
            style=("bold reverse" if selected else "red" if self.failed else ""),
            no_wrap=True,
            overflow="ellipsis",
        )


def grid(cards, columns):
    table = Table.grid(expand=True, padding=(0, 1), pad_edge=False)
    for _ in range(columns):
        table.add_column(ratio=1)
    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        table.add_row(*row, *([""] * (columns - len(row))))
    return table


@dataclass
class TaskPages:
    anchor: str | None = None
    index: int = 0
    pending: int = 0
    bounds: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    columns: int = 1
    order: list = field(default_factory=list)
    scope: str = "unfinished"
    query: str = ""

    def cycle_scope(self):
        scopes = ("unfinished", "all", "history")
        self.scope = (
            scopes[(scopes.index(self.scope) + 1) % len(scopes)]
            if self.scope in scopes
            else "unfinished"
        )
        self.pending = 0

    @property
    def scope_label(self):
        return {
            "unfinished": "未完成/失败",
            "all": "全部",
            "history": "历史",
            "failed": "仅失败",
            "queued": "排队优先",
        }[self.scope]

    def filtered(self, pairs, *, paged=True):
        from smartsom.telemetry.runtime import FAILURES, FINAL

        ordered = self.ordered(pairs)
        selected = []
        for pair in ordered:
            row = pair[1]
            matches = (
                self.query.casefold()
                in (row["id"] + " " + row.get("name", "")).casefold()
            )
            in_scope = (
                self.scope == "all"
                or (self.scope == "history" and row["status"] in FINAL)
                or (self.scope == "failed" and row["status"] in FAILURES)
                or (
                    self.scope == "queued"
                    and row["status"]
                    in {"queued", "pending", "waiting", "waiting_resources"}
                )
                or (
                    self.scope == "unfinished"
                    and (row["status"] not in FINAL or row["status"] in FAILURES)
                )
            )
            if not paged or (matches and in_scope):
                selected.append(pair)
        ids = [row["id"] for _, row in selected]
        self.ids = ids
        if not ids:
            self.bounds, self.index, self.pending = [], 0, 0
        if self.anchor not in ids:
            old_index = (
                self.order.index(self.anchor) if self.anchor in self.order else 0
            )
            self.anchor = next(
                (key for key in self.order[old_index:] if key in ids),
                ids[-1] if ids else None,
            )
        return selected

    def ordered(self, pairs):
        by_id = {row["id"]: (entry, row) for entry, row in pairs}
        self.order = [key for key in self.order if key in by_id] + [
            key for key in by_id if key not in self.order
        ]
        return [by_id[key] for key in self.order]

    def move(self, direction):
        self.pending += direction

    def choose(self, console, ids, cards, *, width, height, columns=None):
        self.columns = columns or min(
            max(1, len(cards)), max(1, (width + GAP) // (MIN_CARD_WIDTH + GAP))
        )
        cell_width = max(1, (width - GAP * (self.columns - 1)) // self.columns)
        row_heights = [
            max(
                len(
                    console.render_lines(
                        card,
                        console.options.update(width=cell_width, height=None),
                        pad=False,
                    )
                )
                for card in cards[start : start + self.columns]
            )
            for start in range(0, len(cards), self.columns)
        ]
        bounds, start, used = [], 0, 0
        for row_index, row_height in enumerate(row_heights):
            offset = row_index * self.columns
            if used and used + row_height > height:
                bounds.append((start, offset))
                start, used = offset, 0
            used += row_height
        if cards:
            bounds.append((start, len(cards)))
        self.bounds, self.ids = bounds, list(ids)
        if not bounds:
            self.index, self.anchor, self.pending = 0, None, 0
            return [], (0, 0)
        if self.anchor in ids:
            anchor_index = ids.index(self.anchor)
            self.index = next(
                i for i, (a, b) in enumerate(bounds) if a <= anchor_index < b
            )
        self.index = max(0, min(len(bounds) - 1, self.index + self.pending))
        moved = bool(self.pending)
        self.pending = 0
        a, b = bounds[self.index]
        if moved or self.anchor not in ids:
            self.anchor = ids[a]
        return cards[a:b], (a, b)

    @property
    def label(self):
        if not self.bounds:
            return "第 1/1 页 · 任务 0/0"
        a, b = self.bounds[self.index]
        return f"第 {self.index + 1}/{len(self.bounds)} 页 · 任务 {a + 1}–{b}/{len(self.ids)} · {self.columns} 列"


def render_panel(
    view,
    *,
    header,
    ids,
    cards,
    compact_cards,
    short_header,
    title,
    subtitle=None,
    footer="",
    border_style=None,
):
    """Measure cells and keep every task reachable in interactive viewports."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    paged = view.paging
    width, height = max(1, view.console.width - 4), max(3, view.console.height - 1)
    columns = min(max(1, len(cards)), max(1, (width + GAP) // (MIN_CARD_WIDTH + GAP)))
    cell_width = max(1, (width - GAP * (columns - 1)) // columns)
    items = list(header)
    notices = (
        [Text(str(view.notice), style="yellow", no_wrap=False, overflow="fold")]
        if view.notice
        else []
    )
    input_mode = getattr(view, "control_input", None)
    if input_mode == "remove":
        footer = "输入目标短 ID · Enter 确认移除 · Esc/Ctrl+C 取消"
    elif input_mode == "limit":
        footer = "输入目标数字 · Enter 确认 · Esc 取消"

    def footer_render(value):
        from rich.cells import cell_len

        lines = []
        for paragraph in value.split("\n"):
            current = ""
            for action in paragraph.split(" · "):
                candidate = current + " · " + action if current else action
                if current and cell_len(candidate) > width:
                    lines.append(current)
                    current = action
                else:
                    current = candidate
            lines.append(current)
        return Text("\n".join(lines), style="dim", no_wrap=False, overflow="fold")

    footer_text = footer_render(footer)
    if paged:
        if width < 90:
            for long, short in (
                ("c 修改总并发", "c 并发"),
                ("d 离开界面", "d 离开"),
                ("安全停止任务", "停止"),
                ("关闭监控", "退出"),
            ):
                footer = footer.replace(long, short)
        footer_text = footer_render(
            (
                ("[ 上页 · ] 下一页 · " if width < 90 else "[ 上一页 · ] 下一页 · ")
                if cards and input_mode is None
                else ""
            )
            + footer,
        )

        def available_space():
            overhead = (
                len(
                    view.console.render_lines(
                        Group(
                            *items,
                            *notices,
                            *([Text("页码")] if cards else []),
                            *([footer_text] if footer else []),
                        ),
                        view.console.options.update(width=width, height=None),
                        pad=False,
                    )
                )
                + 2
            )
            return height - overhead

        available = available_space()
        required = 1 if cards else 0
        if height <= 12 or available < (4 if cards else 0):
            items = list(short_header)
            available = available_space()
        if available < required:
            items = []
            available = available_space()
            if available < required and notices:
                return Panel(
                    Group(*notices),
                    title=title,
                    subtitle=subtitle,
                    box=box.SQUARE,
                    border_style=border_style or "",
                    padding=(0, 1),
                    height=height,
                )
            available = max(required, available)

    if paged and cards:

        def measured(card):
            return len(
                view.console.render_lines(
                    card,
                    view.console.options.update(width=cell_width, height=None),
                    pad=False,
                )
            )

        if any(measured(card) > available for card in cards):
            cards = compact_cards
        if any(measured(card) > available for card in cards):
            from smartsom.telemetry.dashboard import task_eta

            cards = [
                Text(
                    identity
                    + " · ETA "
                    + task_eta(
                        view,
                        view.tasks.get(identity, {"id": identity, "status": "queued"}),
                    ),
                    no_wrap=True,
                    overflow="ellipsis",
                )
                for identity in ids
            ]
        visible, _ = view.task_pages.choose(
            view.console, ids, cards, width=width, height=available
        )
        items += [
            *notices,
            grid(visible, view.task_pages.columns),
            Text(
                "实验卡片 · " + view.task_pages.label,
                style="bold",
                no_wrap=True,
                overflow="ellipsis",
            ),
            footer_text,
        ]
    else:
        items.extend(notices)
        if cards:
            items += [
                Text(f"实验卡片 · 全部 {len(cards)} 个任务", style="bold"),
                grid(cards, columns),
            ]
        if footer:
            items.append(footer_text)
    return Panel(
        Group(*items),
        title=title,
        subtitle=subtitle,
        box=box.SQUARE,
        border_style=border_style or "",
        padding=(0, 1),
        height=height if paged else None,
    )


def runtime_progress(fraction, *, cells=8, compact=False):
    """A fixed-width determinate gauge; circles encode quarters, never animation."""
    from rich.text import Text

    value = Text(no_wrap=True, overflow="ellipsis")
    if fraction is None:
        value.append("○   —" if compact else "─" * cells + "    —", "dim")
        return value
    fraction = min(1, max(0, fraction))
    percent = f"{int(fraction * 100):3}%"
    if compact:
        quadrant = (
            4
            if fraction == 1
            else min(3, max(1, round(fraction * 4)))
            if fraction
            else 0
        )
        value.append("○◔◑◕●"[quadrant], "cyan")
    else:
        complete = int(fraction * cells)
        value.append("━" * complete, "cyan")
        value.append("─" * (cells - complete), "dim")
    value.append(" " + percent)
    return value


def runtime_task_table(rows, *, width, selected=None):
    """Stable columns, a quiet selection arrow and bounded progress gauges."""
    from rich import box
    from rich.text import Text

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        pad_edge=False,
        padding=0,
        collapse_padding=True,
        show_edge=False,
        header_style="bold cyan",
        row_styles=["", "on #18222f"],
    )
    specs = [
        ("marker", "", 1),
        (
            "number",
            "#",
            max(2, max((len(str(row["number"])) for row in rows), default=2)),
        ),
        ("status", "状态", 7),
        ("name", "任务名称", None),
    ]
    if width >= 72:
        specs += [("algorithm", "算法", 4), ("seed", "seed", 5)]
    if width >= 100:
        specs += [
            ("stages" if width >= 140 else "stage", "阶段", 24 if width >= 140 else 6),
            ("phase_progress", "阶段进度", 8),
        ]
    else:
        specs.append(
            ("phase_compact", "阶段", 9 if width >= 72 else 11 if width >= 52 else 6)
        )
    specs += [
        (
            "progress",
            "总进度",
            23 if width >= 140 else 15 if width >= 100 else 13 if width >= 72 else 6,
        )
    ]
    if width >= 52:
        specs.append(("eta", "ETA", 8))
    if width >= 100:
        specs += [("cpu", "CPU核", 5), ("memory", "RAM GiB", 7)]
    if width >= 140:
        specs += [("elapsed", "耗时", 8), ("speed", "速率", 10)]
    if width >= 190:
        specs.append(("warning", "异常 / 提示", 26))
    for key, heading, size in specs:
        table.add_column(
            heading,
            width=size,
            ratio=1 if size is None else None,
            min_width=8 if size is None else None,
            no_wrap=True,
            overflow="ellipsis",
            justify="right" if key in {"number", "cpu", "memory"} else "left",
        )
    for row in rows:
        focused = row["id"] == selected
        cells = []
        for key, _, _ in specs:
            value = (
                ("›" if focused else " ")
                if key == "marker"
                else row.get(key, row["id"] if key == "name" else "—")
            )
            if key == "progress" and "fraction" in row:
                value = runtime_progress(
                    row["fraction"],
                    cells=18 if width >= 140 else 10 if width >= 100 else 8,
                    compact=width < 72,
                )
            style = (
                "not bold #9eacb8"
                if key == "marker"
                else row.get("status_style", "")
                if key == "status"
                else "red"
                if key == "warning" and row.get("warning")
                else ""
            )
            cell = (
                value.copy()
                if isinstance(value, Text)
                else Text(str(value), style=style, no_wrap=True, overflow="ellipsis")
            )
            cells.append(cell)
        table.add_row(*cells, style="on #24496b" if focused else None)
    return table


def runtime_shortcuts(width, *, editing=False, queue=False, controlling=False):
    """Highlight keys independently from their quieter action labels."""
    from rich.text import Text

    actions = (
        [("Enter", "确认"), ("Esc/Ctrl+C", "取消")]
        if editing
        else [("↑↓", "选择"), ("Enter", "加入/取消"), ("←→", "排序"), ("Esc", "返回")]
        if queue and controlling
        else [
            ("↑↓", "选择"),
            ("Enter", "详情"),
            *([("Q", "优先")] if controlling else []),
            *([("/", "搜索"), ("f", "视图")] if width >= 52 else []),
            ("?", "帮助"),
        ]
    )
    footer = Text(no_wrap=True, overflow="ellipsis")
    for index, (key, label) in enumerate(actions):
        if index:
            footer.append("  " if width >= 72 or editing else " ")
        footer.append(key, style="bold #b8e4ff on #263b4b")
        footer.append((" " if width >= 72 or editing else "") + label)
    return footer


def render_runtime_panel(
    view,
    *,
    header,
    short_header,
    ids,
    cards,
    table_rows,
    total_tasks,
    title,
    subtitle,
    feedback,
    reason,
):
    """Keep a dense comparison table above fixed feedback and navigation rows."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    if not view.paging:
        return Panel(
            Group(*header, reason, *feedback, *cards),
            title=title,
            subtitle=subtitle,
            box=box.SQUARE,
        )
    width, height = max(1, view.console.width - 4), max(3, view.console.height - 1)

    def fixed(value, style=""):
        return Text(str(value), style=style, no_wrap=True, overflow="ellipsis")

    editing = getattr(view, "control_input", None) in {"limit", "search"}
    queue = getattr(view, "runtime_queue", False)
    footer = runtime_shortcuts(
        width, editing=editing, queue=queue, controlling=view.controlling
    )
    top = list(short_header)
    # Table header + rule; two message rows, navigation and key strip.
    available = height - 2 - len(top) - 6
    if available < 1:
        top = [short_header[0]]
        available = height - 2 - len(top) - 6
    _, (a, b) = view.task_pages.choose(
        view.console,
        ids,
        [fixed(key) for key in ids],
        width=width,
        height=max(1, available),
        columns=1,
    )
    table = runtime_task_table(
        table_rows[a:b], width=width, selected=view.task_pages.anchor
    )
    selected = next(
        (row for row in table_rows if row["id"] == view.task_pages.anchor), None
    )
    context = (
        f"选中 {selected['id']}"
        + (f" · {selected['warning']}" if selected.get("warning") else "")
        if selected
        else "当前视图无任务 · f 切换全部/历史 · / 修改搜索"
    )
    notice = view.notice or getattr(view, "selection_notice", None)
    receipts = list(feedback)
    if len(receipts) > 1 and str(receipts[1]):
        context = str(receipts[0])
        message = receipts[1]
        if "用时" in str(message) and "已用" not in str(message):
            message = fixed(
                str(message).removeprefix("状态以实际观测为准 · ") + " · " + str(reason)
            )
    else:
        message = fixed("") if str(reason).endswith("未记录") else reason
    if queue:
        priority = (
            (view.tuning or {}).get("runtime_control", {}).get("priority_tasks", [])
        )
        context = "优先队列 · " + (" → ".join(priority) if priority else "默认调度顺序")
    if notice:
        context = str(notice)
    if not ids and not editing:
        context = "当前视图无任务 · f 切换全部/历史 · / 修改搜索"
    hidden = total_tasks - len(ids)
    page = f"第 {view.task_pages.index + 1}/{max(1, len(view.task_pages.bounds))} 页"
    scope = view.task_pages.scope_label
    pagination = f"视图：{scope} · 总{total_tasks} 隐藏{hidden}"
    if view.task_pages.query:
        pagination += f" · 搜索 {view.task_pages.query}"
    navigation = Table.grid(expand=True, padding=(0, 1), pad_edge=False)
    navigation.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    navigation.add_column(justify="right", no_wrap=True)
    navigation.add_row(
        fixed(pagination, "cyan"),
        fixed(f"{a + 1 if ids else 0}–{b}/{len(ids)} · {page}", "dim"),
    )
    return Panel(
        Group(
            *top,
            table,
            *[fixed("") for _ in range(max(0, available - (b - a)))],
            Rule(
                fixed(
                    context, "red" if selected and selected.get("warning") else "dim"
                ),
                align="left",
                style="dim",
            ),
            message,
            navigation,
            footer,
        ),
        title=title,
        subtitle=subtitle,
        box=box.SQUARE,
        padding=(0, 1),
        height=height,
    )
