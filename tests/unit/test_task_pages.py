"""Public Rich pagination stays local and preserves the complete task ledger."""

import io
import json
import os
import re
import select
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from rich.console import Console

from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay


@pytest.mark.parametrize("width", [40, 60, 80, 120, 160, 220])
def test_terminal_svg_preserves_cjk_cells_styles_and_capture_title(width):
    from rich.cells import cell_len
    from rich.text import Text

    from smartsom.telemetry.capture import SVG, cell_svg, export_terminal_svg

    console = Console(file=io.StringIO(), width=width, record=True)
    sample = Text("运行进度 ", style="bold cyan")
    sample.append("CPU 1.2 核 ◑ 42% e\u0301 <&>")
    console.print(sample)
    svg = export_terminal_svg(console, title="SmartSOM · 很长的截图名称" * 20)
    assert cell_svg(svg, columns=width) == svg
    root = ET.fromstring(svg)
    glyphs = [
        node
        for node in root.iter(f"{{{SVG}}}text")
        if "textLength" in node.attrib and not node.get("class", "").endswith("-title")
    ]
    assert (
        "".join(node.text or "" for node in glyphs).replace("\u00a0", " ").rstrip()
        == sample.plain
    )
    assert not any(node.text == "\u0301" for node in glyphs)
    assert "@font-face" not in svg and "https://cdnjs" not in svg
    assert "full-width" not in svg
    assert "font-weight: bold" in svg
    bounds = {}
    for node in glyphs:
        x, length = float(node.get("x")), float(node.get("textLength"))
        row = node.get("y")
        assert x >= bounds.get(row, 0) - 0.001
        assert x + length <= width * 12.2 + 0.001
        assert length == pytest.approx(12.2 * cell_len(node.text))
        assert node.get("lengthAdjust") == "spacingAndGlyphs"
        bounds[row] = x + length
    title = next(
        node
        for node in root.iter(f"{{{SVG}}}text")
        if node.get("class", "").endswith("-title")
    )
    left = float(title.get("x")) - float(title.get("textLength")) / 2
    assert left >= 90
    assert title.text.endswith("…")
    # Capturing does not consume the recording or change live console output.
    assert export_terminal_svg(console, title="same") == export_terminal_svg(
        console, title="same"
    )
    assert console.export_text().rstrip() == sample.plain


def summary_text(value):
    return re.sub(r" +", " ", value).replace("/ ", "/")


def view_at(width=80, height=24, count=37, *, kind="tune"):
    view = RuntimeDisplay(DisplayOptions(verbose=False), kind=kind, readonly=True)
    view.console = Console(file=io.StringIO(), width=width, height=height)
    view.paging = True
    view.stage = "training" if kind == "tune" else "stage_10"
    for index in range(count):
        view.tasks[f"task-{index:03}"] = {
            "id": f"task-{index:03}",
            "name": f"task-{index:03}",
            "stage": "running",
            "status": "running",
            "completed": 3,
            "total": 100,
            "unit": "physical ticks" if kind == "tune" else "entries",
            "values": {
                "evaluation_finished": 3,
                "evaluation_requested": 5,
                "workflow": {"mode": "evaluation"},
            }
            if kind != "tune"
            else {},
        }
    if kind == "tune":
        view.configure_tuning(
            {
                "stage": "training",
                "manual": True,
                "entries": [],
                "calibration": {"level": "off"},
                "concurrency_limit": 2,
            }
        )
    return view


def text(view):
    return "\n".join(
        "".join(segment.text for segment in line)
        for line in view.console.render_lines(
            view.render(), view.console.options.update(height=None), pad=False
        )
    )


@pytest.mark.parametrize("kind", ["tune", "batch-directory"])
@pytest.mark.parametrize(
    "width,height,columns",
    [(60, 12, 1), (80, 24, 1), (110, 24, 2), (160, 36, 3), (250, 45, 4), (400, 45, 7)],
)
def test_every_task_is_reachable_in_readable_columns_and_height_pages(
    kind, width, height, columns
):
    view = view_at(width, height, kind=kind)
    before = json.dumps(view.snapshot(), sort_keys=True)
    seen = []
    while True:
        rendered = text(view)
        assert len(rendered.splitlines()) <= height - 1
        assert "下一页" in rendered and "任务 " in rendered
        assert "Ctrl+C×2" in rendered
        assert view.task_pages.columns == columns
        a, b = view.task_pages.bounds[view.task_pages.index]
        visible = view.task_pages.ids[a:b]
        assert all(identity in rendered for identity in visible)
        seen.extend(visible)
        if view.task_pages.index == len(view.task_pages.bounds) - 1:
            break
        view.move_task_page(1)
    assert seen == list(view.tasks)
    assert json.dumps(view.snapshot(), sort_keys=True) == before
    view.move_task_page(100)
    text(view)
    assert view.task_pages.index == len(view.task_pages.bounds) - 1
    view.move_task_page(-100)
    text(view)
    assert view.task_pages.index == 0


def test_page_anchor_survives_completion_new_tasks_and_terminal_resize():
    view = view_at(80, 24)
    text(view)
    view.move_task_page(2)
    text(view)
    anchor, order = view.task_pages.anchor, list(view.task_pages.ids)
    view.tasks[anchor]["status"] = "completed"
    view.tasks = dict(reversed(list(view.tasks.items())))
    view.tasks["new-task"] = {
        **view.tasks[anchor],
        "id": "new-task",
        "name": "new-task",
        "status": "queued",
    }
    for width, height in [(160, 36), (60, 12), (250, 45), (80, 24)]:
        view.console = Console(file=io.StringIO(), width=width, height=height)
        text(view)
        a, b = view.task_pages.bounds[view.task_pages.index]
        assert anchor in view.task_pages.ids[a:b]
        assert view.task_pages.anchor == anchor
        assert view.task_pages.ids == order + ["new-task"]
    del view.tasks[anchor]
    text(view)
    assert view.task_pages.anchor in view.task_pages.ids
    assert view.task_pages.index < len(view.task_pages.bounds)


@pytest.mark.parametrize("kind", ["tune", "batch-directory"])
@pytest.mark.parametrize("height", [7, 8, 12])
@pytest.mark.parametrize(
    "notice",
    ["总并发上限：_（Enter 确认）", "invalid literal for int()", "再次 Ctrl+C 确认"],
)
def test_short_view_keeps_control_notice(kind, height, notice):
    view = view_at(80, height, kind=kind)
    view.notice = notice
    rendered = text(view)
    assert notice in rendered
    assert rendered.count(notice) == 1
    assert len(rendered.splitlines()) <= height - 1


@pytest.mark.parametrize("kind", ["tune", "batch-directory"])
@pytest.mark.parametrize("stage", ["calibration", "recommended"])
@pytest.mark.parametrize("height", [4, 7, 12])
@pytest.mark.parametrize("notice", ["invalid literal for int()", "再次 Ctrl+C 确认"])
def test_short_view_without_task_cards_keeps_notice(kind, stage, height, notice):
    view = view_at(80, height, count=0, kind=kind)
    view.stage = stage
    if view.tuning:
        view.tuning["stage"] = stage
    view.notice = notice
    rendered = text(view)
    assert notice in rendered
    assert rendered.count(notice) == 1
    assert len(rendered.splitlines()) <= height - 1


@pytest.mark.parametrize("count", [0, 37])
def test_short_view_keeps_stale_warning_and_active_control_prompt(count):
    view = view_at(80, 7, count=count)
    warning = "Last recorded update 7s ago; process state unknown"
    prompt = "总并发上限：_（Enter 确认）"
    view.notice = warning + "\n" + prompt
    rendered = text(view)
    assert warning in rendered and prompt in rendered
    assert len(rendered.splitlines()) <= 6


def cli_environment():
    return {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }


def saved_snapshot(tmp_path):
    view = view_at(count=37)
    view.paging = False
    root = tmp_path / "queue"
    (root / "logs").mkdir(parents=True)
    (root / "logs/progress.json").write_text(json.dumps(view.snapshot()))
    return root


@pytest.mark.parametrize("format", ["text", "json"])
def test_public_once_and_redirected_output_include_all_tasks_without_paging(
    tmp_path, format
):
    root = saved_snapshot(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "monitor",
            str(root),
            "--once",
            "--log-format",
            format,
        ],
        capture_output=True,
        env=cli_environment(),
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert all(f"task-{i:03}" in result.stderr for i in range(37))
    assert "\x1b" not in result.stderr and "下一页" not in result.stderr
    if format == "json":
        assert len(json.loads(result.stderr)["tasks"]) == 37
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.skipif(os.name != "posix", reason="POSIX public PTY keyboard")
@pytest.mark.parametrize("command", ["monitor", "attach"])
@pytest.mark.parametrize("runtime_cards", [False, True])
def test_public_pty_brackets_page_and_detach_without_control_writes(
    tmp_path, command, runtime_cards
):
    import fcntl
    import pty
    import termios

    root = saved_snapshot(tmp_path)
    if runtime_cards:
        path = root / "logs/progress.json"
        snapshot = json.loads(path.read_text())
        snapshot["tuning"]["runtime_control"] = {
            "status": "running",
            "limit": 2,
            "active": 1,
        }
        for index, row in enumerate(snapshot["tasks"]):
            row["status"] = "completed" if index < 20 else "running"
        path.write_text(json.dumps(snapshot))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 12, 80, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "smartsom.experiments.cli", command, str(root)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env={
            **cli_environment(),
            "TERM": "xterm-256color",
            "COLUMNS": "80",
            "LINES": "12",
        },
    )
    os.close(slave)
    output = bytearray()

    def until(expected):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output.extend(os.read(master, 65536))
                plain = re.sub(
                    r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.decode(errors="replace")
                )
                if expected in plain:
                    return
            assert process.poll() is None, output.decode(errors="replace")
        pytest.fail(output.decode(errors="replace"))

    try:
        until("第 1/")
        if runtime_cards:
            until("task-020")
            for label in ["全部", "历史", "未完成/失败"] * 2:
                output.clear()
                os.write(master, b"f")
                until("视图：" + label)
            for width, height in [(160, 50), (60, 24), (80, 12)]:
                output.clear()
                fcntl.ioctl(
                    master, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0)
                )
                os.kill(process.pid, signal.SIGWINCH)
                until("task-020")
        os.write(master, b"]")
        until("第 2/")
        output.clear()
        os.write(master, b"[")
        until("第 1/")
        if command == "attach":
            output.clear()
            os.write(master, b"c")
            until("输入总并发上限")
            output.clear()
            os.write(master, b"\r")
            until("请输入大于 0 的整数")
            os.write(master, b"\x1b")
            until("? 帮助" if runtime_cards else "下一页")
        output.clear()
        os.kill(process.pid, signal.SIGINT)
        until("再次 Ctrl+C")
        os.write(master, b"d")
        # Live.stop prints the complete final summary. Drain the PTY while
        # waiting so its small output buffer cannot block a correct detach.
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    os.read(master, 65536)
                except OSError:
                    break
        assert process.wait(timeout=max(0.01, deadline - time.monotonic())) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.skipif(os.name != "posix", reason="POSIX public PTY output")
@pytest.mark.parametrize("native", [False, True])
def test_redirected_input_with_tty_output_keeps_all_tasks(tmp_path, native):
    import fcntl
    import pty
    import termios

    root = saved_snapshot(tmp_path)
    path = root / "logs/progress.json"
    snapshot = json.loads(path.read_text())
    if native:
        snapshot["kind"] = "batch-directory"
        snapshot["tuning"] = {
            "native_stage": True,
            "batch_training_active": False,
            "entries": [],
            "resources": {
                "cpus_used": 4,
                "cpus_total": 8,
                "cpus_effective": 2,
                "owned_cpu_cores": 2,
            },
            "throughput": {"status": "ok", "total": 12, "training": 0},
            "total_ticks_per_second": 12,
            "training_ticks_per_second": 0,
        }
    snapshot.update(status="completed", stage="completed")
    path.write_text(json.dumps(snapshot))
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 12, 80, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "smartsom.experiments.cli", "monitor", str(root)],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env={
            **cli_environment(),
            "TERM": "xterm-256color",
            "COLUMNS": "80",
            "LINES": "12",
        },
    )
    os.close(slave)
    output = bytearray()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    break
            elif process.poll() is not None:
                break
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)
    plain = output.decode(errors="replace")
    assert all(f"task-{index:03}" in plain for index in range(37))
    assert "下一页" not in plain
    if native:
        assert "CPU" in plain and "12" in plain and "ticks/s" in plain


@pytest.mark.parametrize("long_ids", [False, True])
def test_narrow_remove_confirmation_keeps_complete_target_and_input(long_ids):
    view = view_at(80, 24, count=4)
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 4,
                "active": 4,
                "mode": "manual",
                "status": "running",
            }
        }
    )
    view.control_input = "remove"
    batch_id = "batch-long-" * 8 if long_ids else "batch-a"
    task_id = "task-long-" * 10 if long_ids else "task-000"
    view.notice = (
        f"移除批次 {batch_id}\n任务 {task_id}\n"
        "状态 running · 安全保存后释放资源，保留文件\n"
        "输入 abc1234567 后 Enter 确认：_ · Esc 取消"
    )
    rendered = text(view)
    assert len(rendered.splitlines()) == 23
    assert "abc1234567" in rendered and "Enter" in rendered and "Esc" in rendered
    unwrapped = "".join(row.strip(" │") for row in rendered.splitlines())
    assert batch_id in unwrapped and task_id in unwrapped
    assert "保留文件" in rendered


@pytest.mark.parametrize("width", [60, 80])
@pytest.mark.parametrize("controlling", [False, True])
def test_unified_footer_keeps_all_context_actions_at_supported_narrow_sizes(
    width, controlling
):
    view = view_at(width, 24, count=12)
    view.controlling = controlling
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 4,
                "active": 4,
                "mode": "manual",
                "status": "running",
            }
        }
    )
    rendered = text(view)
    assert len(rendered.splitlines()) == 23
    actions = (
        ["↑↓ 选择", "Enter 详情", "? 帮助", "/ 搜索", "f 视图"]
        if width >= 80
        else ["↑↓选择", "Enter详情", "?帮助", "/搜索", "f视图"]
    )
    for action in actions:
        assert action in rendered
    view.runtime_help = True
    help_text = text(view)
    view.runtime_help_offset = 100000
    help_text += text(view)
    assert "Ctrl+C×2" in help_text
    if controlling:
        for action in [
            "+/- 并发",
            "c 数值",
            "空格 全局暂停/继续",
            "P 暂停选中",
            "R 恢复选中",
            "D 移除确认",
            "停止批次",
        ]:
            assert action in help_text
    else:
        assert "关闭监控" in help_text and "D 移除确认" not in help_text
    view.runtime_help = False
    before = [i for i, row in enumerate(rendered.splitlines()) if "task-" in row]
    bounds = list(view.task_pages.bounds)
    view.runtime_details = True
    after = text(view)
    assert before == [i for i, row in enumerate(after.splitlines()) if "task-" in row]
    assert bounds == view.task_pages.bounds
    footer_y = next(i for i, row in enumerate(after.splitlines()) if "Enter" in row)
    for query in ("task-000", "no-match"):
        view.task_pages.query = query
        filtered = text(view)
        assert len(filtered.splitlines()) == 23
        assert (
            next(i for i, row in enumerate(filtered.splitlines()) if "Enter" in row)
            == footer_y
        )
        assert "暂无请求" not in filtered


@pytest.mark.parametrize("width", [60, 80])
@pytest.mark.parametrize("input_mode", ["limit", "remove"])
def test_input_footer_exposes_confirm_cancel_without_inactive_shortcuts(
    width, input_mode
):
    view = view_at(width, 24, count=4)
    view.controlling = True
    view.control_input = input_mode
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 4,
                "active": 4,
                "mode": "manual",
                "status": "running",
            }
        }
    )
    view.notice = (
        "输入 abc1234567 后 Enter 确认 · Esc 取消"
        if input_mode == "remove"
        else "总并发上限：6（Enter 确认，Esc 取消）"
    )
    rendered = text(view)
    assert len(rendered.splitlines()) == 23
    assert "Enter" in rendered and "Esc" in rendered
    assert "d 离开" not in rendered and "j/k 选择" not in rendered
    if input_mode == "remove":
        assert "abc1234567" in rendered and "Esc/Ctrl+C 取消" in rendered


@pytest.mark.parametrize("width,height", [(60, 24), (80, 24), (160, 50)])
@pytest.mark.parametrize("count", [1, 4, 12])
def test_runtime_table_and_details_keep_task_and_batch_facts(width, height, count):
    view = view_at(width, height, count=count)
    view.controlling = True
    view.overview = {"work_completed": 100, "work_total": 200}
    view.configure_tuning(
        {
            "resources": {
                "owned_cpu_cores": 1.5,
                "owned_memory_bytes": 2 * 1024**3,
                "cpus_used": 99,
            },
            "runtime_control": {
                "limit": 4,
                "active": 2,
                "mode": "adaptive",
                "status": "running",
            },
        }
    )
    for row in view.tasks.values():
        row.update(completed=100, total=100, stage="saving")
        row["values"] = {
            "algorithm": "DQN",
            "updates": 12,
            "workflow": {"training_seed": 7},
        }
    rendered = text(view)
    assert len(rendered.splitlines()) == height - 1
    for fact in [
        "完成 0/",
        "批次 CPU 1.5",
        "RAM 2.00 GiB",
        "自适应",
        "保存",
        "任务进度",
        "ETA",
        "状态",
    ]:
        assert fact in summary_text(rendered)
    assert "运行中" in rendered
    if width >= 80:
        assert "DQN" in rendered and "seed" in rendered
    assert len(view.task_pages.bounds) == 1
    task_rows = [
        row for row in rendered.splitlines() if "task-" in row and "选中" not in row
    ]
    assert len(task_rows) == count
    assert "CPU 99" not in rendered
    assert view.task_pages.ids == list(view.tasks)
    assert any("›" in row and "task-000" in row for row in rendered.splitlines())
    anchor = view.task_pages.ids[-1]
    view.task_pages.anchor = anchor
    text(view)
    view.console = Console(
        file=io.StringIO(), width=80 if width == 60 else 60, height=24
    )
    text(view)
    assert view.task_pages.anchor == anchor
    before = list(view.task_pages.bounds)
    view.runtime_detail_task = anchor
    view.runtime_detail_offset = 0
    details = text(view)
    assert "实验详情" in details and "Esc 返回" in details and anchor in details
    assert view.task_pages.anchor == anchor and view.task_pages.bounds == before
    for offset in range(0, 60, 5):
        view.runtime_detail_offset = offset
        details += text(view)
    for fact in [
        "update 12",
        "argmax: 未记录",
        "DQN",
        "seed 7",
        "主机可用",
        "工作流 100/200",
        "ETA 估算中",
    ]:
        assert fact in " ".join(details.split())
    view.runtime_detail_offset = 100000
    assert "逻辑预算" in text(view) or "budget" in text(view)
    view.runtime_detail_task = None
    assert "Enter" in text(view) and "详情" in text(view)


def test_posix_arrows_decode_without_becoming_page_keys():
    from smartsom.telemetry.monitor import _key

    read_fd, write_fd = os.pipe()
    try:
        for sequence, expected in [
            (b"\x1b[A", "up"),
            (b"\x1b[B", "down"),
            (b"\x1b[C", "right"),
            (b"\x1b[D", "left"),
            (b"\x1b", "\x1b"),
            (b"\x1b[5~", "pageup"),
            (b"\x1b[6~", "pagedown"),
            ("任务"[0].encode(), "任"),
        ]:
            os.write(write_fd, sequence)
            assert _key(read_fd, 0.1) == expected
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_runtime_observed_validation_return_and_sampled_resources_are_explicit():
    view = view_at(60, 24, count=1)
    view.tasks["task-000"]["values"] = {
        "validation_results": {
            "argmax": {"mean_return": -2.25, "update": 3, "return_samples": 2}
        }
    }
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 1,
                "active": 1,
                "mode": "manual",
                "status": "running",
                "tasks": {
                    "task-000": {
                        "status": "running",
                        "resources": {
                            "cpu_cores": 9,
                            "memory": 2 * 1024**3,
                            "memory_peak": 1024**3,
                            "status": "current",
                            "observed_at": 123,
                        },
                    }
                },
            }
        }
    )
    rendered = text(view)
    assert "资源观测已过期" in rendered
    view.runtime_detail_task = "task-000"
    view.runtime_detail_offset = 0
    rendered = text(view)
    for offset in range(0, 40, 5):
        view.runtime_detail_offset = offset
        rendered += text(view)
    normalized = " ".join(rendered.split())
    assert "argmax: -2.25@u3" in normalized
    assert "sampling: 未记录" in normalized and "CPU 未记录" in normalized
    assert "已采样内存峰值" in rendered
    assert "秒前更新" in rendered and "epoch" not in rendered


@pytest.mark.parametrize("width", [36, 56, 76, 156])
@pytest.mark.parametrize("selected", [None, "trial-3"])
def test_table_tasks_remain_separate_and_readable_without_selection(width, selected):
    from smartsom.telemetry.task_pages import runtime_task_table

    rows = [
        {
            "number": index + 1,
            "id": f"trial-{index}",
            "status": "运行中",
            "experiment": f"DQN/{index}",
            "fraction": 0.42,
            "phase_compact": "训练 ·  42%",
            "phase_narrow": "训练 42%",
            "stage": "训练",
            "phase_progress": "42%",
            "cpu": "1.2",
            "memory": "1.00",
            "elapsed": "90s",
            "warning": "",
        }
        for index in range(12)
    ]
    console = Console(file=io.StringIO(), width=width)
    lines = console.render_lines(
        runtime_task_table(rows, width=width, selected=selected)
    )
    task_lines = [line for line in lines if "trial-" in "".join(s.text for s in line)]
    assert len(task_lines) == 12
    for index, segments in enumerate(task_lines):
        value = "".join(segment.text for segment in segments)
        assert value.count("trial-") == 1
        assert f"trial-{index}" in value and "运行中" in value and "42%" in value
        assert ("›" in value) == (f"trial-{index}" == selected)
        assert str(index + 1) in value.split(f"trial-{index}")[0]
    if selected is None:
        backgrounds = [
            {
                segment.style.bgcolor
                for segment in line
                if segment.style and segment.style.bgcolor
            }
            for line in task_lines
        ]
        assert backgrounds[0] != backgrounds[1]
        assert backgrounds[0::2] == [backgrounds[0]] * 6
        assert backgrounds[1::2] == [backgrounds[1]] * 6


@pytest.mark.parametrize("width", [60, 80])
def test_runtime_table_ordinals_survive_filtering_and_progress_fits(width):
    view = view_at(width, 24, count=12)
    view.configure_tuning({"runtime_control": {"status": "running"}})
    for row in view.tasks.values():
        row.update(stage="sampling", completed=42, total=100)
    before = text(view)
    assert len(view.task_pages.bounds) == 1
    for line in before.splitlines():
        if "task-" in line and "选中" not in line:
            assert "42%" in line and "训练" in line
    view.tasks["task-000"]["status"] = "completed"
    view.tasks = dict(reversed(list(view.tasks.items())))
    filtered = text(view)
    assert "总12 隐藏1" in filtered and "task-000" not in filtered
    last = next(line for line in filtered.splitlines() if "task-011" in line)
    assert "12" in last.split("task-011")[0]
    view.task_pages.scope = "all"
    restored = text(view)
    assert "总12 隐藏0" in restored and "已完成" in restored
    assert view.task_pages.ids == [f"task-{index:03}" for index in range(12)]


@pytest.mark.parametrize("width,height", [(60, 24), (80, 24), (160, 50)])
def test_runtime_filter_before_pages_preserves_history_detail_and_anchor(width, height):
    view = view_at(width, height, count=12)
    view.configure_tuning(
        {"runtime_control": {"status": "running", "limit": 2, "active": 1}}
    )
    for index, row in enumerate(view.tasks.values()):
        row["status"] = (
            "completed" if index < 9 else ["running", "paused", "failed"][index - 9]
        )
    rendered = text(view)
    assert view.task_pages.ids == ["task-009", "task-010", "task-011"]
    assert (
        "task-009" in summary_text(rendered)
        and "完成 9/12" in summary_text(rendered)
        and "失败 1" in summary_text(rendered)
    )
    assert view.task_pages.anchor == "task-009"
    view.runtime_detail_task = "task-009"
    view.tasks["task-009"]["status"] = "completed"
    detail = text(view)
    assert "task-009" in detail and "已完成" in detail
    assert view.runtime_detail_task == "task-009"
    view.runtime_detail_task = None
    text(view)
    assert view.task_pages.anchor == "task-010"
    view.task_pages.cycle_scope()
    text(view)
    assert view.task_pages.ids == list(view.tasks)
    assert view.task_pages.anchor == "task-010"
    view.task_pages.cycle_scope()
    text(view)
    assert view.task_pages.ids == [f"task-{i:03}" for i in range(10)] + ["task-011"]
    view.task_pages.cycle_scope()
    text(view)
    view.tasks["new"] = {**view.tasks["task-010"], "id": "new", "status": "pausing"}
    for w, h in [(160, 50), (60, 24), (80, 24)]:
        view.console = Console(file=io.StringIO(), width=w, height=h)
        text(view)
        assert view.task_pages.ids == ["task-010", "task-011", "new"]
        assert view.task_pages.anchor in view.task_pages.ids
        view.move_task_page(100)
        text(view)
        assert view.task_pages.index == len(view.task_pages.bounds) - 1


@pytest.mark.parametrize(
    "status", ["completed", "paused", "pausing", "queued", "stopping"]
)
def test_runtime_filter_empty_and_held_queues(status):
    view = view_at(count=4)
    view.configure_tuning({"runtime_control": {"status": "running"}})
    text(view)  # Populate a previous page before it becomes empty.
    for row in view.tasks.values():
        row["status"] = status
    rendered = text(view)
    if status == "completed":
        assert not view.task_pages.ids and view.task_pages.anchor is None
        assert "当前视图无任务" in rendered
        view.task_pages.cycle_scope()
        text(view)
    assert view.task_pages.ids == list(view.tasks)
    view.paging = False
    view.task_pages.scope = "history"
    rendered = text(view)
    assert all(key in rendered for key in view.tasks)


@pytest.mark.parametrize(
    "width,height", [(40, 12), (60, 20), (60, 24), (80, 24), (160, 40)]
)
def test_runtime_selection_status_and_pages_survive_transient_notices(width, height):
    view = view_at(width, height, count=37)
    view.controlling = True
    view.configure_tuning(
        {"runtime_control": {"status": "running", "limit": 2, "active": 1}}
    )
    view.tasks["task-001"]["status"] = "failed"
    view.tasks["task-001"]["reason"] = "Disk full: release storage before retry"
    view.tasks["task-002"]["status"] = "paused"
    view.task_pages.anchor = "task-001"
    first = text(view)
    bounds = list(view.task_pages.bounds)
    assert len(bounds) > 1
    for notice in ["命令 limit-1: received", None, "请求被拒绝：上限必须大于零"]:
        view.notice = notice
        rendered = text(view)
        assert len(rendered.splitlines()) == height - 1
        assert view.task_pages.bounds == bounds
        assert "失败" in rendered
        assert any("›" in line and "task-001" in line for line in rendered.splitlines())
        assert "ETA 已结束" not in rendered
    if (width, height) == (80, 24):
        assert "task-003" in first  # At least four tasks, instead of eleven pages.
        assert len(bounds) < 12
    view.move_task_page(1)
    rendered = text(view)
    assert any(
        "›" in line and view.task_pages.anchor in line for line in rendered.splitlines()
    )


def test_search_and_failure_filter_do_not_hide_noninteractive_records():
    view = view_at(count=6)
    view.configure_tuning({"runtime_control": {"status": "running"}})
    view.tasks["task-001"].update(status="failed", name="工厂甲 PPO")
    view.tasks["task-003"].update(status="failed", name="工厂乙 DQN")
    snapshot = json.dumps(view.snapshot(), sort_keys=True)
    view.task_pages.scope = "failed"
    view.task_pages.query = "工厂甲"
    assert "task-001" in text(view)
    assert view.task_pages.ids == ["task-001"]
    view.task_pages.query = "missing"
    assert "当前视图无任务" in text(view)
    assert view.task_pages.anchor is None
    view.paging = False
    rendered = text(view)
    assert all(key in rendered for key in view.tasks)
    assert json.dumps(view.snapshot(), sort_keys=True) == snapshot


@pytest.mark.parametrize("mode", ["limit", "search"])
def test_tiny_input_keeps_confirmation_and_cancel_instructions(mode):
    view = view_at(40, 12, count=12)
    view.configure_tuning({"runtime_control": {"status": "running"}})
    view.control_input = mode
    rendered = text(view)
    assert "Enter 确认" in rendered and "Esc/Ctrl+C 取消" in rendered
    assert "↑/↓ 选择" not in rendered


def test_runtime_ordinal_width_is_reserved_across_three_digit_pages():
    from rich.cells import cell_len

    view = view_at(80, 24, count=120)
    view.configure_tuning(
        {"runtime_control": {"limit": 1, "active": 1, "status": "running"}}
    )
    positions = []
    for identity in ["task-000", "task-110", "task-000"]:
        view.task_pages.anchor = identity
        rendered = text(view)
        selected = next(
            line for line in rendered.splitlines() if "›" in line and identity in line
        )
        positions.append(cell_len(selected[: selected.index(identity)]))
    assert len(set(positions)) == 1
