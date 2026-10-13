"""Public Rich pagination stays local and preserves the complete task ledger."""

import io
import json
import os
import re
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console

from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay


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
def test_public_pty_brackets_page_and_detach_without_control_writes(tmp_path, command):
    import fcntl
    import pty
    import termios

    root = saved_snapshot(tmp_path)
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
        os.write(master, b"]")
        until("第 2/")
        output.clear()
        os.write(master, b"[")
        until("第 1/")
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
