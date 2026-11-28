"""Unified run metadata, display selection, W&B forwarding and monitor summary."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from smartsom.config.base_diagnostic import TrackingOptions
from smartsom.config.codec import ConfigurationError
from smartsom.config.experiment_layout import entrypoints
from smartsom.config.reservation import compile_reservation
from smartsom.config.run_meta import (
    META_FILENAME,
    META_SCHEMA,
    RunMeta,
    build_auto_summary,
    load_batch_meta,
    merge_meta,
    placeholder_eta,
    prompt_missing_meta,
    resolved_meta,
    select_display_text,
)
from smartsom.experiments.author_batch import compile_directory
from smartsom.experiments.cli import main
from smartsom.experiments.launch import compare
from smartsom.experiments.reservation import allocate, arguments
from smartsom.experiments.reservation_batch import allocate_directory
from smartsom.telemetry.experiment_tracking import ExperimentTracking
from smartsom.telemetry.monitor_summary import collect_summary, render_summary_text
from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/examples/small/default/experiment.yaml"


def _write_train_experiment(folder: Path, *, seed=101, meta=None, name="exp"):
    src = yaml.safe_load(SOURCE.read_text())
    src["scenario"] = str((SOURCE.parent / src["scenario"]).resolve())
    src["algorithm"] = str((SOURCE.parent / src["algorithm"]).resolve())
    src["train_episodes"] = 100
    src["train_seeds"] = [seed]
    src["drain_frozen"] = True
    run = src.setdefault("run", {})
    run["task"] = "train"
    run["seed"] = seed
    run["seconds"] = None
    run["output"] = str(folder / "out" / name)
    src["tracking"] = {"mode": "disabled"}
    if meta is not None:
        src["meta"] = meta
    entry = folder / name / "experiment.yaml"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(yaml.safe_dump(src, sort_keys=False, allow_unicode=True))
    return entry


def test_run_meta_validation_and_fallbacks():
    with pytest.raises(ValueError, match="single-line"):
        RunMeta(short_title="a5\nline2")
    with pytest.raises(ValueError, match="nonempty"):
        RunMeta(short_description="   ")
    with pytest.raises(ValueError, match="at most"):
        RunMeta(short_title="x" * 33)

    meta = RunMeta(
        short_title="a5",
        full_description="第一行摘要\n第二行详细研究说明",
        tags=("k4", "k4", "five-agv"),
    )
    assert meta.tags == ("k4", "five-agv")

    payload = resolved_meta(
        meta,
        fallback_title="dir-name",
        auto_summary="small · dqn/base · 1000ep · s101-104",
    )
    # Raw omitted fields remain null, never "N/A".
    assert payload["short_title"] == "a5"
    assert payload["long_title"] is None
    assert payload["short_description"] is None
    assert payload["full_description"] == "第一行摘要\n第二行详细研究说明"
    assert payload["effective"]["short_title"] == "a5"
    assert payload["effective"]["long_title"] == "a5"
    assert payload["effective"]["short_description"] == "第一行摘要"
    assert select_display_text(payload, "short_title") == "a5"
    assert select_display_text(payload, "combined") == "a5"
    assert select_display_text(payload, "none") is None
    assert select_display_text(payload, "auto_summary") == (
        "small · dqn/base · 1000ep · s101-104"
    )


def test_meta_does_not_alter_scientific_identity_or_matrix(tmp_path):
    before = compile_reservation(SOURCE)
    after = compile_reservation(
        SOURCE,
        meta={
            "short_title": "a5",
            "long_title": "K4 五车全角色学习对比",
            "short_description": "K4 五车，全角色学习",
            "full_description": "多行说明\n记录动机",
            "tags": ["k4"],
        },
    )
    assert after.scientific_sha256 == before.scientific_sha256
    assert after.meta.short_title == "a5"
    assert after.facts["task"] == "smoke"
    assert build_auto_summary(after.facts)

    # Compare must see zero scientific differences between two runs differing only in meta.
    r1 = allocate(before, root=tmp_path / "r1")
    r2 = allocate(after, root=tmp_path / "r2")
    diffs = compare(r1, r2)
    assert not any(row["field"].startswith("scientific.") for row in diffs)

    # Matrix over experiment.meta.* is forbidden as a scientific axis.
    doc = yaml.safe_load(SOURCE.read_text())
    doc["scenario"] = str((SOURCE.parent / doc["scenario"]).resolve())
    doc["algorithm"] = str((SOURCE.parent / doc["algorithm"]).resolve())
    doc["matrix"] = {"experiment.meta.short_title": ["a1", "a2"]}
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ConfigurationError, match="invalid scientific matrix axis"):
        from smartsom.experiments.launch import compile_launch

        compile_launch(matrix_path)


def test_batch_meta_yaml_ignored_by_entrypoints_and_merges_by_precedence(tmp_path):
    batch_dir = tmp_path / "k4-study"
    _write_train_experiment(
        batch_dir,
        seed=101,
        name="a",
        meta={"short_description": "子实验覆盖简介", "tags": ["member"]},
    )
    _write_train_experiment(batch_dir, seed=102, name="b")
    (batch_dir / META_FILENAME).write_text(
        yaml.safe_dump(
            {
                "schema": META_SCHEMA,
                "short_title": "a5",
                "long_title": "K4 五车全角色学习对比",
                "short_description": "K4 五车，全角色学习",
                "tags": ["batch"],
            },
            allow_unicode=True,
        )
    )
    found = entrypoints(batch_dir)
    assert all(p.name != META_FILENAME for p in found)
    assert len(found) == 2

    loaded = load_batch_meta(batch_dir)
    assert loaded is not None and loaded.short_title == "a5"

    plan = compile_directory(
        batch_dir,
        meta={"long_title": "CLI 覆盖长标题", "tags": ["cli"]},
    )
    assert plan.meta.short_title == "a5"
    assert plan.meta.long_title == "CLI 覆盖长标题"
    assert plan.meta.short_description == "K4 五车，全角色学习"
    assert list(plan.meta.tags) == ["batch", "cli"]

    by_name = {row["id"]: row["compiled"].meta for row in plan.files}
    assert by_name["a"].short_description == "子实验覆盖简介"
    assert by_name["b"].short_description == "K4 五车，全角色学习"
    assert "member" in by_name["a"].tags and "batch" in by_name["a"].tags

    root = allocate_directory(plan)
    frozen_plan = json.loads((root / "plan.json").read_text())
    frozen_run = json.loads((root / "run.json").read_text())
    assert frozen_plan["meta"]["short_title"] == "a5"
    assert frozen_run["meta"]["short_title"] == "a5"
    assert frozen_plan["tracking_group"].startswith("a5-")
    assert frozen_plan["facts"]["seeds"]["all"] == [101, 102]


def test_cli_requires_meta_for_formal_runs_unless_provided_or_skipped(
    tmp_path, monkeypatch, capsys
):
    entry = _write_train_experiment(tmp_path / "formal", seed=101, name="one")
    launched = []

    def fake_run_launch(plan, **kwargs):
        launched.append(plan)
        return {"status": "completed"}

    monkeypatch.setattr("smartsom.experiments.launch.run_launch", fake_run_launch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

    # Missing meta in non-interactive formal train fails with clear guidance.
    rc = main(["run", str(entry)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "short_title" in err and "short_description" in err
    assert "--allow-empty-meta" in err
    assert not launched

    # Providing CLI flags succeeds and attaches meta to plan.
    rc = main(
        [
            "run",
            str(entry),
            "--short-title",
            "a5",
            "--short-description",
            "K4 五车，全角色学习",
            "--long-title",
            "K4 五车全角色学习对比",
        ]
    )
    assert rc == 0
    assert len(launched) == 1
    assert launched[-1].meta.short_title == "a5"
    assert launched[-1].meta.long_title == "K4 五车全角色学习对比"

    # Explicit --allow-empty-meta bypasses enforcement without writing "N/A".
    rc = main(["run", str(entry), "--allow-empty-meta"])
    assert rc == 0
    assert launched[-1].meta.short_title is None
    assert launched[-1].resolved_meta()["short_title"] is None
    assert launched[-1].resolved_meta()["auto_summary"]

    # Smoke runs never require meta.
    rc = main(["run", str(SOURCE)])
    assert rc == 0


def test_interactive_prompt_collects_missing_meta_and_retries_blank():
    answers = iter(["", "a5", "  ", "K4 五车，全角色学习"])
    buf = io.StringIO()
    facts = {
        "task_count": 4,
        "selected_task_count": 4,
        "seeds": {"selected": [101, 102], "all": [101, 102]},
        "scales": ["small"],
        "algorithms": ["dqn"],
        "observations": ["base"],
        "tasks": ["train"],
        "budget": {"train_episodes": 1000},
        "concurrency": {"max_concurrent": 2, "mode": "manual"},
        "tracking": {"mode": "online"},
    }
    completed = prompt_missing_meta(
        RunMeta(long_title="保留原长标题"),
        facts,
        input_fn=lambda: next(answers),
        output=buf,
    )
    assert completed.short_title == "a5"
    assert completed.short_description == "K4 五车，全角色学习"
    assert completed.long_title == "保留原长标题"
    assert "输入无效" in buf.getvalue()
    assert "SmartSOM 启动确认" in buf.getvalue()


def test_wandb_forwards_meta_notes_name_tags_and_config(tmp_path, monkeypatch):
    calls = []

    class FakeRun:
        url = "https://wandb.ai/team/SmartSOM/runs/abc"
        entity = "team"
        project = "SmartSOM"
        sync_dir = str(tmp_path / "sync")

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

        def finish(self, *args, **kwargs):
            pass

    def fake_init(**kwargs):
        calls.append(kwargs)
        return FakeRun()

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Settings=lambda **kwargs: kwargs,
        teardown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    plan = compile_reservation(
        SOURCE,
        meta={
            "short_title": "a5",
            "long_title": "K4 五车全角色学习对比",
            "short_description": "K4 五车，全角色学习",
            "full_description": "完整研究说明\n第二行",
            "tags": ["k4"],
        },
    )
    root = allocate(plan, root=tmp_path / "run-a5")
    args, _ = arguments(root)
    manifest = {
        "source": {"commit": "abc", "implementation_sha256": "def"},
        "inputs": {},
        "seed": 101,
        "tracking_group": args.tracking_group,
        "experiment_short_name": args.experiment_short_name,
        "meta": args.run_meta,
        "facts": args.run_facts,
    }
    tracker = ExperimentTracking(
        root,
        TrackingOptions(
            mode="offline", name_field="short_title", notes_field="full_description"
        ),
        job_type="train",
    )
    tracker.start(manifest, {"train_seeds": [101], "run": {"output": str(root)}})
    tracker.close()

    assert len(calls) == 1
    init_kwargs = calls[0]
    assert init_kwargs["name"].startswith("a5－")
    assert init_kwargs["group"].startswith("a5-")
    assert init_kwargs["notes"] == "完整研究说明\n第二行"
    assert "title:a5" in init_kwargs["tags"]
    assert "k4" in init_kwargs["tags"]
    assert any(tag.startswith("host:") for tag in init_kwargs["tags"])
    assert init_kwargs["config"]["meta"]["short_title"] == "a5"
    assert init_kwargs["config"]["facts"]["host"]["name"]

    # Configurable notes_field="none" omits notes and name_field="long_title" switches prefix.
    calls.clear()
    tracker2 = ExperimentTracking(
        root,
        TrackingOptions(mode="offline", name_field="long_title", notes_field="none"),
        job_type="train",
    )
    tracker2.start(manifest, {"train_seeds": [101], "run": {"output": str(root)}})
    tracker2.close()
    assert calls[0]["name"].startswith("K4 五车全角色学习对比－")
    assert "notes" not in calls[0]


def test_runtime_display_binds_meta_and_renders_configured_fields(tmp_path):
    from rich.console import Console

    from smartsom.telemetry.tuning_dashboard import runtime_detail, runtime_render

    plan = compile_reservation(
        SOURCE,
        meta={
            "short_title": "b6a",
            "long_title": "机器菜单缩窄对照",
            "short_description": "机器菜单缩窄 (EDD/MDD/CR/SLACK)",
            "full_description": "详细实验背景",
        },
    )
    root = allocate(plan, root=tmp_path / "b6a-run")
    display = RuntimeDisplay(
        DisplayOptions(title_field="combined", description_field="short_description"),
        kind="batch-directory",
        console=Console(width=120, height=30, record=True, file=io.StringIO()),
        quiet=True,
    )
    display.bind(root)
    snap = json.loads((root / "logs/progress.json").read_text())
    assert snap["meta"]["short_title"] == "b6a"
    assert snap["facts"]["task"] == "smoke"
    assert snap["eta"]["schema"] == placeholder_eta()["schema"]

    display.tuning = {
        "runtime_control": {
            "status": "running",
            "active": 1,
            "limit": 2,
            "mode": "manual",
            "tasks": {"t1": {"status": "running"}},
        }
    }
    display.tasks = {
        "t1": {"id": "t1", "name": "t1", "status": "running", "values": {}}
    }
    wide = Console(width=120, height=30, record=True, file=io.StringIO())
    display.console = wide
    wide.print(runtime_render(display))
    rendered = wide.export_text()
    assert "b6a · 机器菜单缩窄对照" in rendered
    assert "机器菜单缩窄 (EDD/MDD/CR/SLACK)" in rendered

    # Narrow terminal degrades combined title to short_title.
    narrow = Console(width=70, height=24, record=True, file=io.StringIO())
    display.console = narrow
    narrow.print(runtime_render(display))
    narrow_text = narrow.export_text()
    assert "SmartSOM · b6a" in narrow_text

    # Detail view includes metadata section.
    detail_console = Console(width=100, height=30, record=True, file=io.StringIO())
    display.console = detail_console
    detail_console.print(
        runtime_detail(display, display.tasks["t1"], {"status": "running"})
    )
    detail_text = detail_console.export_text()
    assert "实验信息" in detail_text
    assert "详细实验背景" in detail_text


def test_monitor_summary_folds_batch_children_and_supports_legacy_runs(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    batch_dir = tmp_path / "inputs" / "a5"
    _write_train_experiment(batch_dir, seed=101, name="s1")
    _write_train_experiment(batch_dir, seed=102, name="s2")
    (batch_dir / META_FILENAME).write_text(
        yaml.safe_dump(
            {
                "schema": META_SCHEMA,
                "short_title": "a5",
                "long_title": "K4 五车全角色学习",
                "short_description": "K4 五车，全角色学习",
            },
            allow_unicode=True,
        )
    )
    plan = compile_directory(
        batch_dir,
        operations={"output_root": str(runs_root / "a5-batch")},
    )
    batch_root = allocate_directory(plan)
    # Simulate live child episode progress & runtime status.
    (batch_root / "control").mkdir(exist_ok=True)
    (batch_root / "control/runtime.json").write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at": 1795774500.0,
                "tasks": {
                    row["id"]: {"status": "running"}
                    for row in json.loads((batch_root / "plan.json").read_text())[
                        "files"
                    ]
                },
            }
        )
    )
    files = json.loads((batch_root / "plan.json").read_text())["files"]
    for idx, row in enumerate(files):
        (batch_root / row["child"] / "progress.json").write_text(
            json.dumps(
                {
                    "status": "training",
                    "episodes": 178 + idx * 4,
                    "episode_budget": 1000,
                    "physical_ticks": 180000,
                }
            )
        )

    # Create a legacy run directory without meta or facts.
    legacy = runs_root / "legacy-run"
    legacy.mkdir(parents=True)
    (legacy / "run.json").write_text(
        json.dumps({"id": "legacy-1", "status": "completed", "stage": "completed"})
    )

    summary = collect_summary([runs_root])
    assert summary["schema"] == "smartsom.monitor-summary/v1"
    # Only 2 top-level entries: batch + legacy (child trial dirs are folded, not duplicated).
    assert len(summary["batches"]) == 2
    assert summary["counts"]["running"] == 1
    assert summary["counts"]["completed"] == 1

    active = summary["batches"][0]
    assert active["meta"]["short_title"] == "a5"
    assert active["meta"]["short_description"] == "K4 五车，全角色学习"
    assert active["progress"]["active_tasks"] == 2
    assert active["progress"]["episodes"] == {"min": 178, "max": 182, "budget": 1000}
    assert active["eta"]["schema"] == "smartsom.eta/v1"
    assert active["eta"]["confidence"] == "unavailable"

    legacy_row = summary["batches"][1]
    assert legacy_row["meta"]["short_title"] is None
    assert legacy_row["meta"]["effective"]["short_title"] == "legacy-run"

    # --active-only excludes completed runs.
    active_only = collect_summary([runs_root], active_only=True)
    assert len(active_only["batches"]) == 1

    # CLI `smartsom runs --summary --json` and text mode both work.
    capsys.readouterr()
    assert main(["runs", "--root", str(runs_root), "--summary", "--json"]) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["schema"] == "smartsom.monitor-summary/v1"
    assert len(cli_payload["batches"]) == 2

    text = render_summary_text(cli_payload)
    assert "a5" in text and "178-182/1000" in text
    assert merge_meta(None, {"short_title": "z1"}).short_title == "z1"


def test_interactive_prompt_preserves_child_meta_overrides(tmp_path, monkeypatch):
    batch_dir = tmp_path / "k4-prompt"
    _write_train_experiment(
        batch_dir,
        seed=101,
        name="a",
        meta={"long_title": "子实验专属长标题"},
    )
    _write_train_experiment(batch_dir, seed=102, name="b")
    (batch_dir / META_FILENAME).write_text(
        yaml.safe_dump(
            {
                "schema": META_SCHEMA,
                "short_title": "a5",
                "long_title": "批次默认长标题",
            },
            allow_unicode=True,
        )
    )
    launched = []
    monkeypatch.setattr(
        "smartsom.experiments.launch.run_launch",
        lambda plan, **kwargs: (launched.append(plan), {"status": "completed"})[1],
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "终端补填简介")

    assert main(["run", str(batch_dir)]) == 0
    plan = launched[0]
    assert plan.meta.short_title == "a5"
    assert plan.meta.short_description == "终端补填简介"
    by_name = {row["id"]: row["compiled"].meta for row in plan.files}
    assert by_name["a"].long_title == "子实验专属长标题"
    assert by_name["a"].short_description == "终端补填简介"
    assert by_name["b"].long_title == "批次默认长标题"


def test_single_active_run_status_and_corrupt_json_resilience(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    plan = compile_reservation(
        SOURCE,
        meta={"short_title": "s1", "short_description": "单实验运行中"},
    )
    single_root = allocate(plan, root=runs_root / "single-active")
    # allocate() leaves batch.json as {"status": "allocated"}; live progress is in progress.json.
    (single_root / "progress.json").write_text(
        json.dumps({"status": "training", "episodes": 40, "episode_budget": 100})
    )
    # Corrupt non-dict JSON in a sibling run must not crash host summary.
    corrupt = runs_root / "corrupt-run"
    corrupt.mkdir(parents=True)
    (corrupt / "run.json").write_text(json.dumps(["not-a-dict"]))

    # Verify psutil sampling primes before final measurement.
    intervals = []

    class FakeVM:
        total = 16 * 1024**3
        available = 8 * 1024**3
        percent = 50.0

    fake_psutil = SimpleNamespace(
        cpu_count=lambda logical=True: 8,
        cpu_percent=lambda interval=None: (intervals.append(interval), 42.5)[1],
        virtual_memory=lambda: FakeVM(),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    summary = collect_summary([runs_root])
    assert summary["counts"]["running"] == 1
    assert summary["host"]["cpu_percent"] == 42.5
    assert len(intervals) == 2
    active = next(b for b in summary["batches"] if b["directory"] == str(single_root))
    assert active["status"] == "training"
    assert active["progress"]["active_tasks"] == 1
    assert active["progress"]["episodes"] == {"min": 40, "max": 40, "budget": 100}


def test_multi_seed_children_freeze_own_seed_and_canonical_single_file_loads_study_meta(
    tmp_path,
):
    from smartsom.experiments.launch import compile_launch

    study = tmp_path / "experiments" / "topic" / "k4-study"
    variant_entry = _write_train_experiment(study, seed=101, name="base")
    doc = yaml.safe_load(variant_entry.read_text())
    doc["train_seeds"] = [101, 102]
    doc["run"].pop("output", None)
    variant_entry.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    (study / META_FILENAME).write_text(
        yaml.safe_dump(
            {
                "schema": META_SCHEMA,
                "short_title": "k4",
                "short_description": "K4 研究批次",
            },
            allow_unicode=True,
        )
    )

    plan = compile_launch(variant_entry)
    assert plan.meta.short_title == "k4"
    assert plan.meta.short_description == "K4 研究批次"
    by_seed = {
        row["compiled"].experiment.run.seed: row["compiled"] for row in plan.files
    }
    assert by_seed[101].facts["seeds"] == {"selected": [101], "all": [101]}
    assert by_seed[102].facts["seeds"] == {"selected": [102], "all": [102]}
    assert "s102" in by_seed[102].resolved_meta()["auto_summary"]

    # Malformed meta.yaml raises ConfigurationError.
    bad_dir = tmp_path / "bad-meta"
    bad_dir.mkdir()
    (bad_dir / META_FILENAME).write_text("schema: wrong/v1\nshort_title: bad\n")
    with pytest.raises(ConfigurationError, match="invalid batch meta.yaml"):
        load_batch_meta(bad_dir)


def test_summary_reports_committed_work_when_wall_budget_ends(tmp_path):
    from smartsom.telemetry.monitor_summary import summarize_run

    child = tmp_path / "trial"
    child.mkdir()
    (tmp_path / "plan.json").write_text(
        json.dumps({"files": [{"id": "a", "child": "trial"}]})
    )
    (tmp_path / "batch.json").write_text(
        json.dumps({"status": "completed", "files": {"a": {"status": "completed"}}})
    )
    (child / "progress.json").write_text(
        json.dumps(
            {"status": "wall_budget_reached", "episode_budget": 100, "episodes": 3}
        )
    )
    for root in (child, tmp_path):
        view = summarize_run(root)["progress"]
        assert view["episodes"] == {"min": 3, "max": 3, "budget": 100}
        assert view["completed_tasks"] == 1
        assert view["fraction"] == 0.03


@pytest.mark.parametrize("previous", ["prepared", "partial"])
def test_summary_prioritizes_newer_failed_launch_over_stale_journal(tmp_path, previous):
    import os

    (tmp_path / "plan.json").write_text('{"files":[]}')
    (tmp_path / "batch.json").write_text('{"status":"failed","stage":"startup"}')
    (tmp_path / "control").mkdir()
    journal = tmp_path / "control/runtime.json"
    journal.write_text(json.dumps({"status": previous}))
    os.utime(journal, (100, 100))
    os.utime(tmp_path / "batch.json", (101, 101))
    view = collect_summary([tmp_path], active_only=True)
    assert view["counts"]["failed"] == 1
    assert view["batches"] == []
    # A new running journal can take ownership during recovery.
    journal.write_text('{"status":"running"}')
    assert collect_summary([tmp_path], active_only=True)["counts"]["running"] == 1


@pytest.mark.parametrize("status", ["smoke", "calibrating", "evaluating"])
def test_active_native_tasks_and_case_progress(tmp_path, status):
    from smartsom.telemetry.monitor_summary import summarize_run

    child = tmp_path / "trial"
    child.mkdir()
    (child / "plan.json").write_text("{}")
    (child / "progress.json").write_text(
        json.dumps({"status": status, "completed_cases": 20, "total_cases": 40})
    )
    (row,) = collect_summary([tmp_path], active_only=True)["batches"]
    assert row["progress"]["active_tasks"] == 1
    if status != "smoke":
        assert row["progress"]["fraction"] == 0.5
        (tmp_path / "plan.json").write_text('{"files":[{"id":"a","child":"trial"}]}')
        assert summarize_run(tmp_path)["progress"]["fraction"] == 0.5


def test_discovery_prunes_owned_and_excluded_directories(tmp_path, monkeypatch):
    import os

    from smartsom.telemetry.monitor_summary import discover_top_level_runs

    child = tmp_path / "study/batch/trials/seed-101"
    child.mkdir(parents=True)
    (child / "manifest.json").touch()
    batch = tmp_path / "study/batch"
    (batch / "plan.json").touch()
    excluded = tmp_path / "tracking"
    (excluded / "sdk").mkdir(parents=True)
    (excluded / "sdk/run.json").touch()
    visited = []
    scandir = os.scandir

    def observe(path):
        visited.append(Path(path))
        return scandir(path)

    monkeypatch.setattr(os, "scandir", observe)
    assert discover_top_level_runs([tmp_path]) == [batch]
    assert visited.count(batch) == 1
    assert not any(
        p.is_relative_to(excluded) or p.is_relative_to(batch / "trials")
        for p in visited
    )


@pytest.mark.parametrize("tune", ["off", "recommend"])
@pytest.mark.parametrize("submit_directory", [False, True])
def test_cli_accepts_study_title_and_variant_description(
    tmp_path, monkeypatch, tune, submit_directory
):
    study = tmp_path / "experiments/topic/study"
    entry = _write_train_experiment(study, meta={"short_description": "variant notes"})
    (study / META_FILENAME).write_text(f"schema: {META_SCHEMA}\nshort_title: study\n")
    doc = yaml.safe_load(entry.read_text())
    doc["run"].update(tune=tune, seconds=120)
    entry.write_text(yaml.safe_dump(doc))
    launched = []

    def launch(plan, **kwargs):
        launched.append(plan)
        return {"status": "completed"}

    monkeypatch.setattr("smartsom.experiments.launch.run_launch", launch)
    assert main(["run", str(entry.parent if submit_directory else entry)]) == 0
    assert launched[0].meta.short_title == "study"
    assert launched[0].meta.short_description == "variant notes"


def test_cli_meta_guard_checks_only_selected_entries(tmp_path, monkeypatch):
    study = tmp_path / "experiments/topic/study"
    _write_train_experiment(study, name="train")
    entry = _write_train_experiment(study, name="smoke")
    doc = yaml.safe_load(entry.read_text())
    doc["run"].update(task="smoke", seconds=120)
    entry.write_text(yaml.safe_dump(doc))
    selected = []

    def launch(plan, **kwargs):
        selected.extend(
            row["compiled"].experiment.run.task
            for row in plan.files
            if row["id"] in plan.selection["tasks"]
        )
        return {"status": "completed"}

    monkeypatch.setattr("smartsom.experiments.launch.run_launch", launch)
    assert main(["run", str(study), "--id", "smoke"]) == 0
    assert selected == ["smoke"]
    assert main(["run", str(study), "--id", "train"]) == 2


def test_selecting_one_named_member_uses_its_own_cli_metadata(tmp_path, monkeypatch):
    study = tmp_path / "experiments/topic/study"
    for name, seed in (("a", 101), ("b", 102)):
        _write_train_experiment(
            study,
            name=name,
            seed=seed,
            meta={"short_title": name, "short_description": f"{name} notes"},
        )
    plans = []

    def launch(plan, **kwargs):
        plans.append(plan)
        return {"status": "completed"}

    monkeypatch.setattr("smartsom.experiments.launch.run_launch", launch)
    assert main(["run", str(study), "--id", "a", "--select", "train_seeds: [101]"]) == 0
    assert plans[0].meta.short_title == "a"
    assert plans[0].meta.short_description == "a notes"


def test_incremental_extension_inherits_meta_and_exports_current_selection(tmp_path):
    from smartsom.experiments.launch import compile_launch
    from smartsom.telemetry.monitor_summary import summarize_run

    study = tmp_path / "experiments/topic/study"
    entry = _write_train_experiment(
        study, name="base", meta={"short_title": "yaml", "tags": ["base-only"]}
    )
    options = {"output_root": str(tmp_path / "runs")}

    def selection(seed, **kwargs):
        return compile_launch(
            study,
            ids=["base"],
            selectors=[{"train_seeds": [seed]}],
            operations=options,
            **kwargs,
        )

    root = allocate_directory(
        selection(
            101, meta={"short_title": "cli", "short_description": "initial notes"}
        )
    )
    original_plan = (root / "plan.json").read_bytes()
    doc = yaml.safe_load(entry.read_text())
    doc["train_seeds"] = [101, 102, 103]
    entry.write_text(yaml.safe_dump(doc))
    allocate_directory(selection(102))
    assert (root / "plan.json").read_bytes() == original_plan
    child = json.loads((root / "trials/base/seed-102/plan.json").read_text())
    assert (child["meta"]["short_title"], child["meta"]["short_description"]) == (
        "cli",
        "initial notes",
    )
    for seed in (102, 103):
        if seed == 103:
            allocate_directory(selection(103))
        view = summarize_run(root)
        assert view["facts"]["task_count"] == 3
        assert view["facts"]["selected_task_count"] == 1
        assert view["facts"]["seeds"] == {"all": [101, 102, 103], "selected": [seed]}
        assert view["meta"]["auto_summary"].endswith(f"s{seed}")

    _write_train_experiment(
        study, name="another", seed=104, meta={"tags": ["another-only"]}
    )
    allocate_directory(
        compile_launch(
            study,
            ids=["another"],
            selectors=[{"train_seeds": [104]}],
            operations=options,
        )
    )
    new_variant = json.loads((root / "trials/another/seed-104/plan.json").read_text())
    assert new_variant["meta"]["tags"] == ["another-only"]
    assert new_variant["meta"]["short_title"] == "cli"


def test_rule_evaluation_facts_follow_executed_methods(tmp_path):
    entry = _write_train_experiment(tmp_path)
    plan = compile_reservation(
        entry, task="evaluate", sets=["experiment.run.methods=[rule,random]"]
    )
    assert plan.facts["algorithms"] == ["rule", "random"]
    assert plan.facts["budget"]["train_episodes"] is None
    assert plan.facts["budget"]["train_ticks"] is None
    summary = plan.resolved_meta()["auto_summary"]
    assert "rule/random" in summary and "ppo" not in summary and "100ep" not in summary


@pytest.mark.parametrize("command", ["monitor", "attach"])
def test_reconnect_respects_frozen_fields_then_cli_overrides(
    tmp_path, monkeypatch, command
):
    root = allocate(
        compile_reservation(
            SOURCE,
            operations={
                "display": {"title_field": "long_title", "description_field": "none"}
            },
        ),
        root=tmp_path / "run",
    )
    options = []

    def monitor(root, **kwargs):
        options.append(kwargs["options"])
        return 0

    monkeypatch.setattr("smartsom.telemetry.monitor.monitor", monitor)
    assert main([command, str(root), "--once"]) == 0
    assert (options[-1].title_field, options[-1].description_field) == (
        "long_title",
        "none",
    )
    assert (
        main([command, str(root), "--once", "--description-field", "full_description"])
        == 0
    )
    assert (options[-1].title_field, options[-1].description_field) == (
        "long_title",
        "full_description",
    )
