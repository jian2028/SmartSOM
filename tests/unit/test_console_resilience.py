"""A lost presentation sink must not change task or evidence outcomes."""

import errno
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from smartsom.telemetry.runtime import (
    CURRENT,
    DisplayOptions,
    RuntimeDisplay,
    _PlainDiagnosticStream,
    operation,
)


class FailingStream(io.StringIO):
    def __init__(self, method="write", error=None):
        super().__init__()
        self.method = method
        self.error = error or OSError(errno.EINVAL, "lost output")
        self.armed = False
        self.failures = 0

    def _fail(self, method):
        if self.armed and method == self.method:
            self.failures += 1
            raise self.error

    def write(self, text):
        self._fail("write")
        return super().write(text)

    def flush(self):
        self._fail("flush")
        return super().flush()

    def isatty(self):
        return True


def view_for(stream, *, live=False, **options):
    console = Console(file=stream, force_terminal=live, width=80, height=24)
    return RuntimeDisplay(
        DisplayOptions(verbose=True, progress="on" if live else "off", **options),
        console=console,
    )


@pytest.mark.parametrize("method", ["write", "flush"])
@pytest.mark.parametrize(
    "error", [OSError(errno.EINVAL, "lost output"), BrokenPipeError()]
)
def test_output_failure_does_not_abort_progress(tmp_path, method, error):
    stream = FailingStream(method, error)
    view = view_for(stream, format="json", debug=True)
    view.bind(tmp_path)
    stream.armed = True
    view.phase("training")
    view.update("learner", {"stage": "sampling", "sampled_steps": 256})
    view.diagnostic({"committed": 256})
    view.finish("completed")
    view.close()
    saved = json.loads((tmp_path / "logs/progress.json").read_text())
    assert saved["status"] == "completed"
    assert saved["tasks"][0]["values"]["sampled_steps"] == 256
    assert saved["console_error"]["type"] == type(error).__name__
    assert "completed" in (tmp_path / "logs/runtime.log").read_text()
    assert json.loads((tmp_path / "logs/debug.jsonl").read_text().splitlines()[-1]) == {
        "committed": 256
    }
    assert stream.failures == 1
    assert not stream.closed
    assert view.console.file is stream


@pytest.mark.parametrize("boundary", ["debug", "backend", "legacy", "live_close"])
def test_each_presentation_boundary_disables_failed_mirror(tmp_path, boundary):
    stream = FailingStream()
    view = view_for(stream, live=boundary == "live_close", debug=True)
    before = sys.stdout, sys.stderr
    view.start()
    view.bind(tmp_path)
    stream.armed = True
    if boundary == "debug":
        view.diagnostic({"message": "retained"})
        assert "retained" in (tmp_path / "logs/debug.jsonl").read_text()
    elif boundary == "backend":
        with (tmp_path / "logs/backend.log").open("w") as file:
            backend = _PlainDiagnosticStream(file, view)
            backend.write("retained backend\n")
            backend.flush()
        assert (tmp_path / "logs/backend.log").read_text() == "retained backend\n"
    elif boundary == "legacy":
        view.legacy_warning()
    if boundary != "live_close":
        view.finish("completed")
    view.close()
    assert stream.failures == 1
    assert view.live is None
    assert (sys.stdout, sys.stderr) == before
    assert not stream.closed
    assert json.loads((tmp_path / "logs/progress.json").read_text())["console_error"]


def test_actual_closed_pipe_preserves_operation_result(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    stream = os.fdopen(write_fd, "w", encoding="utf-8", buffering=1)
    console = Console(file=stream, force_terminal=False)
    monkeypatch.setattr("smartsom.telemetry.runtime.Console", lambda **kwargs: console)

    @operation("run")
    def work():
        CURRENT.get().bind(tmp_path)
        (tmp_path / "work-completed").write_text("yes")
        return SimpleNamespace(status="completed")

    try:
        assert work().status == "completed"
        assert (tmp_path / "work-completed").read_text() == "yes"
        saved = json.loads((tmp_path / "logs/progress.json").read_text())
        assert saved["status"] == "completed"
        assert saved["console_error"]["errno"] in {errno.EINVAL, errno.EPIPE}
    finally:
        # The caller owns this pipe, including its possible buffered close error.
        try:
            stream.close()
        except OSError:
            pass


@pytest.mark.parametrize("filename", ["progress.tmp", "runtime.log", "debug.jsonl"])
def test_evidence_write_failures_remain_fatal(tmp_path, monkeypatch, filename):
    stream = FailingStream()
    view = view_for(stream, debug=True)
    view.bind(tmp_path)
    original = Path.open

    def fail(path, *args, **kwargs):
        if path.name == filename:
            raise OSError(errno.ENOSPC, "disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError, match="disk full"):
        if filename == "debug.jsonl":
            view.diagnostic({"message": "must persist"})
        else:
            view.publish(force=True)
    assert view._console_stream.failure is None


def test_backend_persistence_and_render_errors_remain_fatal(tmp_path, monkeypatch):
    stream = FailingStream()
    stream.armed = True
    view = view_for(io.StringIO())
    backend = _PlainDiagnosticStream(stream, view)
    with pytest.raises(OSError):
        backend.write("authoritative backend output")
    view.bind(tmp_path)
    monkeypatch.setattr(
        view, "text_summary", lambda: (_ for _ in ()).throw(OSError("render bug"))
    )
    with pytest.raises(OSError, match="render bug"):
        view.publish(force=True)
    assert view._console_stream.failure is None


def test_task_exception_survives_failed_error_mirror(tmp_path, monkeypatch):
    stream = FailingStream()
    console = Console(file=stream, force_terminal=False)
    monkeypatch.setattr("smartsom.telemetry.runtime.Console", lambda **kwargs: console)
    original = ValueError("task failed")

    @operation("run")
    def work():
        CURRENT.get().bind(tmp_path)
        stream.armed = True
        raise original

    with pytest.raises(ValueError) as raised:
        work()
    assert raised.value is original
    saved = json.loads((tmp_path / "logs/progress.json").read_text())
    assert saved["status"] == "failed"
    assert saved["notice"] == "task failed"
    assert stream.failures == 1
