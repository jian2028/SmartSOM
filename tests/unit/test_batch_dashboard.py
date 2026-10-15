"""Parent batch progress remains live across rule and training stages."""

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from smartsom.experiments.author_batch import _publish, _run_tune
from smartsom.telemetry.runtime import CURRENT, RuntimeDisplay
from smartsom.telemetry.timeline import _calibration, _formal


def test_parent_projects_rule_case_progress_into_three_stage_dashboard(tmp_path):
    child = tmp_path / "experiments/control_zero"
    (child / "entries/entry-0001/logs").mkdir(parents=True)
    (child / "batch.json").write_text(
        json.dumps({"entries": {"entry-0001": {"status": "running"}}})
    )
    progress = child / "entries/entry-0001/logs/progress.json"
    progress.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "values": {
                            "evaluation_finished": 3,
                            "evaluation_requested": 5,
                            "evaluation_case_active": True,
                            "evaluation_tick": 1941,
                            "evaluation_tick_limit": 4096,
                        }
                    }
                ]
            }
        )
    )
    plan = {
        "files": [
            {
                "id": "control_zero",
                "task": "evaluate",
                "stage": 10,
                "child": "experiments/control_zero",
                "entry_ids": ["entry-0001"],
            }
        ]
    }
    state = {
        "stage": "stage_10",
        "files": {"control_zero": {"status": "running"}},
    }
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    view.total_tasks = 1
    view.preflight = {"status": "passed", "checks_done": 1, "checks_total": 1}
    token = CURRENT.set(view)
    try:
        _publish(tmp_path, plan, state)
    finally:
        CURRENT.reset(token)
    values = view.tasks["control_zero"]["values"]
    assert values["evaluation_finished"] == 3
    assert values["evaluation_tick"] == 1941
    assert _formal(view, "formal")[1] == (3 + 1941 / 4096) / 5
    output = StringIO()
    console = Console(file=output, width=110, color_system=None)
    console.print(view.render())
    text = output.getvalue()
    assert "整体流程" in text
    assert "规则对照" in text
    assert "3/5" in text
    assert "1941/4096" in text
    assert "当前阶段" in text


def test_batch_monitor_heartbeat_refreshes_without_task_state_change():
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.updated_at = 100.1
    before = view._frame_state()
    view.updated_at = 101.1
    assert view._frame_state() != before
    ordinary = RuntimeDisplay(kind="run", quiet=True, readonly=True)
    ordinary.updated_at = 100.1
    before = ordinary._frame_state()
    ordinary.updated_at = 101.1
    assert ordinary._frame_state() == before


@pytest.mark.parametrize("task", ["evaluate", "train-evaluate"])
def test_directory_projects_source_execution_contract_for_rule_and_tune(tmp_path, task):
    from smartsom.telemetry.execution_metrics import source_execution

    identity = "file__entry" if task == "train-evaluate" else "file"
    observation = {
        "schema": "smartsom.execution-observation/v1",
        "attempt": "actual-worker-attempt",
        "sequence": 2,
        "updated_at": 100,
        "training_ticks": 10 if task == "train-evaluate" else 0,
        "execution_ticks": 20,
        "phases": {},
    }
    child = tmp_path / ("performance/tune" if task == "train-evaluate" else "child")
    (child / "logs").mkdir(parents=True)
    (child / "batch.json").write_text(
        json.dumps({"entries": {identity: {"status": "running"}}})
    )
    snapshot = {
        "tasks": [{"id": identity, "execution": observation, "values": {}}],
        "execution": {"aggregate": observation},
    }
    (child / "logs/progress.json").write_text(json.dumps(snapshot))
    plan = {
        "files": [
            {
                "id": "file",
                "task": task,
                "stage": 0,
                "child": "child",
                "entry_ids": ["entry"],
            }
        ]
    }
    state = {
        "stage": "stage_0",
        "files": {"file": {"status": "running"}},
        **({"tune_directory": "performance/tune"} if task == "train-evaluate" else {}),
    }
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    token = CURRENT.set(view)
    try:
        _publish(tmp_path, plan, state)
    finally:
        CURRENT.reset(token)
    assert view.tasks[identity]["execution"] == observation
    aggregate = source_execution(view.snapshot())
    assert aggregate["execution_ticks"] == 20
    assert aggregate["producer_attempts"] == ["actual-worker-attempt"]


def test_preflight_does_not_mark_scheduled_calibration_complete():
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.preflight = {
        "status": "running",
        "checks_done": 8,
        "checks_total": 10,
    }
    view.tuning = {
        "stage": "preflight",
        "calibration_status": "queued",
        "calibration": {"level": "quick"},
    }
    assert _calibration(view, "preflight")[0] == "未开始"
    output = StringIO()
    Console(file=output, width=90, color_system=None).print(view.render())
    text = output.getvalue()
    assert "性能评估" in text and "未开始" in text
    assert "实验执行" in text
    assert "规则对照＋训练评估" not in text


def test_explicit_calibration_skip_is_distinct_from_completion():
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.tuning = {
        "stage": "stage_10",
        "calibration_status": "skipped",
        "calibration": {"level": "off"},
    }
    assert _calibration(view, "formal")[0] == "跳过"
    output = StringIO()
    Console(file=output, width=90, color_system=None).print(view.render())
    assert "性能评估" not in output.getvalue()
    assert "实验执行" in output.getvalue()


@pytest.mark.parametrize(
    "level,expected", [("online", [False]), ("off", [False]), ("quick", [True, False])]
)
def test_directory_online_execution_has_no_separate_recommendation_stage(
    tmp_path, monkeypatch, level, expected
):
    from contextlib import nullcontext

    from smartsom.experiments import author_batch, batch

    plan = {
        "files": [
            {
                "id": "learning",
                "stage": 10,
                "task": "train-evaluate",
                "entry_ids": ["entry-0001"],
            }
        ]
    }
    state = {
        "status": "prepared",
        "stage": "prepared",
        "tune_directory": "tune",
        "calibration_status": level
        if level == "online"
        else "skipped"
        if level == "off"
        else "queued",
        "files": {"learning": {"status": "queued"}},
    }
    calls = []
    monkeypatch.setattr(author_batch, "load", lambda *_: (tmp_path, plan, state))
    monkeypatch.setattr(author_batch, "bind", lambda *_: None)
    # This scheduling test replaces process ownership; admission is tested separately.
    from smartsom.experiments import queue_admission

    monkeypatch.setattr(queue_admission, "initialize", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue_admission, "smoke_slot", lambda *args: nullcontext())
    monkeypatch.setattr(author_batch, "_save", lambda *_: None)
    monkeypatch.setattr(author_batch, "_publish", lambda *_: None)
    monkeypatch.setattr(author_batch, "_smoke_all", lambda *_: None)
    monkeypatch.setattr(author_batch, "requested", lambda *_: False)
    monkeypatch.setattr(batch, "exclusive_lock", lambda *_: nullcontext())

    def execute(*_args, recommend_only, **_kwargs):
        calls.append(recommend_only)
        return {"entries": {"learning__entry-0001": {"status": "completed"}}}

    monkeypatch.setattr(author_batch, "_run_tune", execute)
    result = author_batch.execute_saved(tmp_path)
    assert result["status"] == "completed"
    assert calls == expected


@pytest.mark.parametrize("width,height", [(60, 12), (80, 24), (160, 36)])
def test_online_directory_rule_stage_retains_execution_performance_card(width, height):
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.console = Console(file=StringIO(), width=width, height=height)
    view.paging = True
    view.stage = "stage_10"
    view.tuning = {"stage": "stage_10", "calibration": {"level": "online"}}
    view.tasks["control"] = {
        "id": "control",
        "name": "control",
        "unit": "entries",
        "status": "running",
        "values": {
            "workflow": {"mode": "evaluation"},
            "evaluation_finished": 3,
            "evaluation_requested": 5,
        },
    }
    lines = view.console.render_lines(
        view.render(), view.console.options.update(height=None), pad=False
    )
    assert len(lines) <= height - 1
    view.console.print(view.render())
    text = view.console.file.getvalue()
    assert "性能评估" not in text and "性能 · 在线调整" in text
    assert "3/5" in text and "实验执行" in text


@pytest.mark.parametrize(
    "parent_status,ledger_status,active_count",
    [
        ("completed", "completed", 0),
        ("failed", "failed", 0),
        ("stopped", "stopped", 0),
        ("stopped", "interrupted", 0),
        ("stage_20", "queued", 1),
        ("stage_20", "running", 1),
    ],
)
def test_directory_online_count_uses_terminal_ledger_and_live_broker(
    tmp_path, parent_status, ledger_status, active_count
):
    from smartsom.telemetry.tuning_dashboard import performance_lines

    tune = tmp_path / "performance/tune"
    (tune / "logs").mkdir(parents=True)
    identity = "train__entry-0001"
    (tune / "batch.json").write_text(
        json.dumps({"entries": {identity: {"status": ledger_status}}})
    )
    (tune / "logs/progress.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": identity,
                        "status": "running",
                        "stage": "sampling",
                        "total": 64,
                        "values": {"physical_ticks": 32},
                    }
                ],
                "tuning": {
                    "stage": "running",
                    "online": {
                        "g": {
                            "active": 1,
                            "limit": 2,
                            "ceiling": 3,
                            "latest_throughput": 10,
                            "measured_concurrency": 1,
                        }
                    },
                    "entries": [
                        {
                            "experiment_id": identity,
                            "calibration_group": "g",
                            "status": "running",
                        }
                    ],
                },
            }
        )
    )
    plan = {
        "calibration_level": "online",
        "files": [
            {
                "id": "train",
                "task": "train-evaluate",
                "stage": 20,
                "entry_ids": ["entry-0001"],
            }
        ],
    }
    state = {
        "stage": parent_status,
        "tune_directory": "performance/tune",
        "calibration_status": "online",
        "files": {"train": {"status": "running" if active_count else parent_status}},
    }
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    token = CURRENT.set(view)
    try:
        _publish(tmp_path, plan, state)
    finally:
        CURRENT.reset(token)
    assert view.tasks[identity]["status"] == ledger_status
    assert view.tuning["entries"][0]["status"] == (
        "running" if active_count else ledger_status
    )
    assert any(
        f"实际运行 {active_count}" in row for row in performance_lines(view.tuning)
    )
    assert any("10 ticks/s" in row for row in performance_lines(view.tuning))
    output = StringIO()
    Console(file=output, width=120, color_system=None).print(view.render())
    assert f"实际运行 {active_count}" in output.getvalue()


def test_interactive_batch_launch_shows_preparation_before_attach(monkeypatch):
    from rich.console import Console

    from smartsom.experiments import author_batch, cli
    from smartsom.telemetry import monitor

    events = []

    class Indicator:
        def __enter__(self):
            events.append("spinner")
            return self

        def __exit__(self, *_args):
            events.append("spinner_closed")

        def update(self, message):
            events.append(message)

    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
            stderr=SimpleNamespace(isatty=lambda: True),
        ),
    )
    monkeypatch.setattr(Console, "status", lambda *_a, **_k: Indicator())
    monkeypatch.setattr(
        author_batch,
        "compile_directory",
        lambda *_a, **_k: events.append("compiled") or object(),
    )
    monkeypatch.setattr(
        author_batch,
        "run",
        lambda *_a, **_k: (
            events.append("launched")
            or {"background": True, "directory": "/tmp/test", "status": "running"}
        ),
    )
    monkeypatch.setattr(monitor, "monitor", lambda *_a, **_k: events.append("attached"))
    assert cli.main(["batch-run", "/tmp/test"]) == 0
    assert events[0:2] == ["spinner", "compiled"]
    assert events.index("spinner_closed") < events.index("attached")


def test_failed_tune_worker_reason_reaches_parent_and_rich(tmp_path, monkeypatch):
    from smartsom.experiments import author_batch

    tune = tmp_path / "performance/tune"
    tune.mkdir(parents=True)
    (tune / "batch.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "entries": {
                    "train__entry-0001": {
                        "status": "failed",
                        "failure": {
                            "message": "remote traceback\nValueError: worker implementation differs from frozen batch"
                        },
                    }
                },
            }
        )
    )
    plan = {
        "files": [
            {
                "id": "train",
                "task": "train-evaluate",
                "stage": 20,
                "entry_ids": ["entry-0001"],
            }
        ]
    }
    state = {
        "status": "failed",
        "stage": "failed",
        "tune_directory": "performance/tune",
        "files": {"train": {"status": "failed"}},
    }

    class Process:
        exitcode = 1

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self):
            pass

    monkeypatch.setattr(
        author_batch.multiprocessing,
        "get_context",
        lambda *_: SimpleNamespace(Process=lambda **_: Process()),
    )
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    token = CURRENT.set(view)
    try:
        _publish(tmp_path, plan, state)
        with pytest.raises(
            RuntimeError,
            match="train__entry-0001: ValueError: worker implementation differs",
        ):
            _run_tune(tmp_path, plan, state, recommend_only=False)
        view.finish("failed", error="ValueError: worker implementation differs")
    finally:
        CURRENT.reset(token)
    assert view.tasks["train__entry-0001"]["reason"].startswith("ValueError:")
    output = StringIO()
    Console(file=output, width=110, color_system=None).print(view.render())
    assert "worker implementation differs" in output.getvalue()


def test_live_queue_rates_ignore_history_and_count_waiting_time():
    from smartsom.experiments.live_queue import Throughput

    rates = Throughput()

    def row(name, ticks):
        return {
            "id": name,
            "unit": "physical ticks",
            "values": {"physical_ticks": ticks},
            "execution": {
                "schema": "smartsom.execution-observation/v1",
                "attempt": name,
                "sequence": ticks,
                "training_ticks": ticks,
                "execution_ticks": ticks,
                "phases": {},
                "updated_at": None,
            },
        }

    assert rates.sample([row("a", 1000)], 10) is None
    assert rates.sample([row("a", 1100), row("b", 5000)], 20) == 10
    assert rates.sample([row("a", 1100), row("b", 5000)], 30) == 5
    assert rates.sample([row("a", 1200), row("b", 5100)], 40) == 10


def test_live_queue_limit_drains_without_preemption():
    from smartsom.experiments.live_queue import admission_jobs

    jobs = {
        name: {
            "id": name,
            "kind": "adopted" if i < 3 else "native",
            "status": "running" if i < 3 else "queued",
        }
        for i, name in enumerate("abcde")
    }
    state = {"limit": 2, "jobs": jobs}
    assert admission_jobs(state, memory_available=64 * 1024**3) == []
    assert all(jobs[name]["status"] == "running" for name in "abc")
    state["limit"] = 4
    assert [
        job["id"] for job in admission_jobs(state, memory_available=64 * 1024**3)
    ] == ["d"]
    state["limit"] = 5
    assert len(admission_jobs(state, memory_available=64 * 1024**3)) == 2
    assert admission_jobs(state, memory_available=1024**3) == []
    assert admission_jobs(state, memory_available=64 * 1024**3, stopping=True) == []


def test_manual_queue_card_shows_measured_cpu_and_throughput():
    from smartsom.telemetry.runtime import RuntimeDisplay
    from smartsom.telemetry.tuning_dashboard import render

    view = RuntimeDisplay(kind="batch-directory")
    view.console = Console(file=StringIO(), width=180, height=60, force_terminal=False)
    view.controlling = True
    view.status = "running"
    view.stage = "training"
    view.configure_tuning(
        {
            "stage": "training",
            "manual": True,
            "queue_open": True,
            "active": 3,
            "concurrency_limit": 5,
            "entries": [],
            "total_ticks_per_second": 42.5,
            "resources": {
                "cpus_used": 3.2,
                "cpus_total": 32,
                "memory_available": 100 * 1024**3,
                "mode": "manual",
            },
            "calibration": {"level": "off", "calibrated": True},
        }
    )
    view.console.print(render(view))
    text = view.console.file.getvalue()
    assert "3/5" in text and "3.2/32" in text and "42.5 ticks/s" in text
    assert "c 修改总并发" in text
    assert "schedule_uncalibrated" not in text


@pytest.mark.parametrize("controlling", [True, False])
def test_queue_keyboard_changes_limit_only_in_attach(
    tmp_path, monkeypatch, controlling
):
    from smartsom.experiments import live_queue
    from smartsom.telemetry import monitor as module
    from smartsom.telemetry.runtime import DisplayOptions

    snapshot = {
        "schema": "smartsom.runtime-progress/v1",
        "name": "queue",
        "kind": "batch-directory",
        "stage": "training",
        "status": "running",
        "updated_at": 100.0,
        "tasks": [],
        "tuning": {"manual": True},
    }
    monkeypatch.setattr(module, "read_snapshot", lambda _: snapshot)
    keys = iter(["[", "]", "c", "]", "5", "[", "\r", "d"])
    monkeypatch.setattr(module, "_key", lambda *_: next(keys))
    changes = []
    monkeypatch.setattr(
        live_queue, "set_limit", lambda root, value: changes.append((root, value))
    )
    module.monitor(
        tmp_path,
        controlling=controlling,
        options=DisplayOptions(verbose=False, progress="off"),
    )
    assert changes == ([(tmp_path, 5)] if controlling else [])


def test_queue_rates_include_case_ticks_without_counting_adopted_history():
    from smartsom.experiments.live_queue import Throughput

    rates = Throughput()

    def row(seed, tick, active=True, training=1000):
        return [
            {
                "id": "a",
                "values": {
                    "physical_ticks": training,
                    "validation_tick": tick,
                    "validation_case_active": active,
                    "data_seed": seed,
                    "validation_round": 1,
                },
            }
        ]

    assert rates.sample(row(101, 500), 0) is None
    assert rates.sample(row(101, 600), 10) is None
    assert rates.payload["reason"] == "legacy source has training-only counters"
    assert rates.training_rate == 0
    assert rates.sample(row(101, 700, False), 20) is None
    assert rates.sample(row(102, 100), 30) is None
    # A new inactive case must not contribute its historical final ticks.
    assert rates.sample(row(103, 4096, False, 1100), 40) is None
    assert rates.training_rate == pytest.approx(
        100 / 30
    )  # exact retained 30-second window


def test_queue_collect_preserves_adopted_allocation(tmp_path):
    import json

    from smartsom.experiments.live_queue import collect

    (tmp_path / "logs").mkdir()
    (tmp_path / "batch.json").write_text(
        json.dumps(
            {
                "status": "running",
                "files": {"seed101": {"status": "running"}},
            }
        )
    )
    (tmp_path / "logs/progress.json").write_text(
        json.dumps(
            {
                "tasks": [{"id": "seed101__entry", "values": {"physical_ticks": 2048}}],
                "tuning": {
                    "entries": [
                        {
                            "experiment_id": "seed101__entry",
                            "status": "running",
                            "threads": 1,
                            "actual_cpus": 1,
                        }
                    ]
                },
            }
        )
    )
    tasks, entries = collect(
        {
            "jobs": {
                "seed101": {
                    "id": "seed101",
                    "kind": "adopted",
                    "root": str(tmp_path),
                    "entry_ids": ["seed101__entry"],
                    "status": "running",
                }
            }
        }
    )
    assert entries[0]["actual_cpus"] == 1
    assert entries[0]["threads"] == 1
    assert entries[0]["experiment_id"] == tasks[0]["id"]
    assert tasks[0]["values"]["physical_ticks"] == 2048


def test_manual_plain_summary_reports_resources_instead_of_calibration():
    from smartsom.telemetry.tuning_dashboard import summary_lines

    text = "\n".join(
        summary_lines(
            {
                "manual": True,
                "active": 3,
                "concurrency_limit": 5,
                "total_ticks_per_second": 42.5,
                "training_ticks_per_second": 0,
                "resources": {"cpus_used": 5, "cpus_total": 32},
            }
        )
    )
    assert "3/5" in text and "5/32" in text and "42.5 ticks/s" in text
    assert "训练 0 ticks/s" in text
    assert "推荐设置" not in text and "性能评估" not in text
