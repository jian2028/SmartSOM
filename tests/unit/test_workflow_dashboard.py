"""The terminal displays actual work units, phase boundaries and seed identities."""

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from smartsom import api
from smartsom.config.experiment import LoggingOptions, prepare
from smartsom.experiments.cli import _parser
from smartsom.telemetry.dashboard import next_line, phase_bar
from smartsom.telemetry.monitor import read_snapshot
from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay
from smartsom.telemetry.workflow import WorkflowWork, describe_config, task_title

ROOT = Path(__file__).resolve().parents[2]


def metadata(mode="study"):
    return {
        "mode": mode,
        "title": "My experiment [literal]",
        "task_title": "{algorithm} H={H} V={V} {travel} Seed {seed}",
        "factory": "Small",
        "map": "19×9",
        "algorithm": "PPO",
        "algorithms": ["PPO", "DQN"],
        "training_seed": 101,
        "training_seeds": [101, 102, 103],
        "validation_seed": 303,
        "evaluation_seed": 202,
        "training_total": 16384,
        "training_unit": "physical ticks",
        "round_size": 256,
        "round_total": 64,
        "validation_every": 4,
        "validation_rounds": 16,
        "validation_cases": 5,
        "validation_horizons": [4096] * 5,
        "evaluation_cases": 5,
        "evaluation_horizons": [4096] * 5,
        "max_concurrent": 8,
        "social_information": "OFF",
    }


def view(width=180, height=50, kind="study", count=8):
    stream = io.StringIO()
    result = RuntimeDisplay(
        DisplayOptions(verbose=False),
        kind=kind,
        console=Console(file=stream, width=width, height=height),
    )
    result.configure(metadata(kind))
    if kind == "study":
        result.total_tasks = 24
    with result.batch_updates():
        for index in range(count):
            result.update(
                str(index),
                {
                    "stage": "sampling",
                    "physical_ticks": 320,
                    "updates": 1,
                    "validation_batches_finished": 0,
                    "study_case": {
                        "algorithm": "PPO",
                        "H": "h0",
                        "V": "low",
                        "travel": "auto",
                        "seed": 101 + index % 3,
                    },
                },
                total=16384,
                unit="physical ticks",
            )
    return result


def plain(console, content):
    output = io.StringIO()
    target = Console(file=output, width=console.width, height=console.height)
    target.print(content)
    return output.getvalue()


@pytest.mark.parametrize("width,height", [(80, 24), (100, 35), (120, 36), (180, 50)])
@pytest.mark.parametrize("count", [1, 8])
def test_terminal_adapts_without_overflow_and_keeps_total_eta(width, height, count):
    result = view(width, height, count=count)
    for stage in ("sampling", "optimizing", "validation", "saving", "evaluation"):
        for row in result.tasks.values():
            row["stage"] = stage
            row["values"].update(
                validation_finished=2,
                validation_requested=5,
                validation_tick=1024,
                validation_tick_limit=4096,
                validation_round=1,
                evaluation_finished=1,
                evaluation_requested=5,
                evaluation_tick=1024,
                evaluation_tick_limit=4096,
            )
        content = result.render()
        lines = result.console.render_lines(
            content, result.console.options.update(height=None), pad=False
        )
        assert len(lines) == height - 1
        assert all(
            sum(segment.cell_length for segment in line) <= width for line in lines
        )
        text = plain(result.console, content)
        assert "总流程" in text and "ETA" in text and "My experiment [literal]" in text
        if width == 180 and count == 8:
            assert all(f"Seed {seed}" in text for seed in (101, 102, 103))


def test_sampling_round_and_validation_boundary_are_distinct():
    result = view(count=1)
    row = result.tasks["0"]
    assert "距第1批验证剩3轮（含当前）" in next_line(result, row).plain
    assert "64/256 physical ticks" in plain(result.console, phase_bar(result, row))
    row["stage"] = "saving"
    row["values"]["updates"] = 4
    assert "第1批验证" in next_line(result, row).plain
    row["values"]["validation_batches_finished"] = 1
    assert "距第2批验证剩4轮（含当前）" in next_line(result, row).plain
    row["values"].update(updates=64, validation_round=16)
    row["stage"] = "validation"
    assert "最终评估" in next_line(result, row).plain


def test_evaluation_does_not_disguise_episodes_as_training_ticks():
    result = view(kind="evaluation", count=0)
    result.update(
        "evaluation",
        {
            "stage": "evaluation",
            "evaluation_finished": 2,
            "evaluation_requested": 5,
            "evaluation_tick": 100,
            "evaluation_tick_limit": 200,
        },
        total=5,
        unit="evaluation episodes",
    )
    result.update(
        "child-run",
        {"stage": "simulation", "physical_ticks": 100},
        total=200,
        unit="physical ticks",
    )
    text = plain(result.console, result.render())
    assert "评估案例" in text and "Evaluation root 202" in text
    assert (
        "训练预算" not in text and "循环已完成" not in text and "child-run" not in text
    )
    assert text.count("评估案例") == 1


def test_budgeted_work_not_actual_tick_sum_and_no_double_case_credit():
    work = WorkflowWork()
    work.started = 0
    plan = metadata("train-evaluate")
    rows = {
        "training": {
            "id": "training",
            "stage": "validation",
            "completed": 1024,
            "values": {
                "validation_batches_finished": 0,
                "validation_finished": 1,
                "validation_tick": 2048,
                "validation_tick_limit": 4096,
                "validation_case_active": True,
            },
        }
    }
    first = work.overview(plan, rows, now=0)
    assert first["work_completed"] == 1024 + 4096 + 2048
    assert first["work_total"] == 16384 + 16 * 5 * 4096 + 5 * 4096
    assert work.overview(plan, rows, now=30)["eta_seconds"] is None
    rows["training"]["values"].update(
        validation_finished=2, validation_case_active=False
    )
    last = work.overview(plan, rows, now=60)
    assert last["work_completed"] == 1024 + 2 * 4096
    assert last["eta_seconds"] > 0
    rows["training"].update(stage="saving")
    rows["training"]["values"].update(validation_batches_finished=1)
    assert work.overview(plan, rows, now=61)["work_completed"] == 1024 + 5 * 4096


@pytest.mark.parametrize(
    "bad", ["{unsupported}", "{id.__class__}", "{seed:>20}", "{seed!r}", "hello\nworld"]
)
def test_titles_reject_hidden_fields_and_control_lines(bad):
    with pytest.raises(ValueError):
        LoggingOptions(task_title=bad)


def test_display_title_is_configurable_and_does_not_change_scientific_identity():
    config = api.load_config(ROOT / "configs/test/runs/template1_train_static.yaml")
    before = prepare(config)
    config.logging.title = "My map comparison"
    config.logging.task_title = "{algorithm} / Seed {seed}"
    after = prepare(config)
    assert after.scientific_sha256 == before.scientific_sha256
    assert task_title("{name}", {"name": "[literal]"}, "default") == "[literal]"
    for args in (
        ["train", "--config", "example.yaml"],
        ["train-evaluate", "--config", "example.yaml"],
        ["run", "--config", "example.yaml"],
        ["evaluate", "RUN"],
    ):
        parsed = _parser().parse_args([*args, "--progress-title", "My title"])
        assert parsed.progress_title == "My title"


def test_legacy_training_units_remain_adapter_decisions():
    plan = describe_config(
        {
            "training": {"total_steps": 100, "steps_per_update": 32},
            "validation": {"enabled": False},
        },
        "training",
    )
    assert plan["training_unit"] == "adapter decisions"
    assert plan["round_total"] == 4 and plan["validation_rounds"] == 0


def test_main_learner_updates_are_not_named_optimizer_minibatches():
    result = view(kind="training", count=0)
    result.update(
        "training",
        {"stage": "learning_metrics", "sampled_steps": 256, "learner_updates": 7},
        total=16384,
        unit="adapter decisions",
    )
    text = plain(result.console, result.render())
    assert "学习器更新 7" in text and "优化步数 7" not in text


@pytest.mark.learning
@pytest.mark.parametrize("constant_returns", [False, True])
def test_main_ppo_emits_validation_and_evaluation_cases_in_one_workflow(
    tmp_path, monkeypatch, constant_returns
):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    if constant_returns:
        from sb3_contrib import MaskablePPO

        original_train = MaskablePPO.train

        def train_constant_targets(model):
            # A finite constant return batch has undefined explained variance
            # even though the actual optimizer update and losses are valid.
            model.rollout_buffer.returns.fill(0)
            return original_train(model)

        monkeypatch.setattr(MaskablePPO, "train", train_constant_targets)
    config = api.load_config(ROOT / "configs/test/runs/learning_sb3.yaml")
    config.training.total_steps = 8
    config.training.steps_per_update = 4
    config.training.max_decisions = 32
    config.training.max_ticks = 8
    config.scenario_overrides["tick_limit"] = 8
    config.algorithm.batch_size = 4
    config.algorithm.n_epochs = 1
    config.algorithm.hidden_sizes = (8, 8)
    config.validation.enabled = True
    config.validation.every_updates = 1
    config.validation.replications = 2
    config.evaluation.replications = 2
    config.evaluation.full_replay = False
    config.evaluation.verbose = False
    config.output.root = str(tmp_path)
    config.logging.tensorboard = False
    config.logging.verbose = False
    config.logging.title = "Main PPO validation and evaluation"

    result = api.train_evaluate(config)
    if constant_returns:
        record = json.loads((result.training.run_dir / "run.json").read_text())
        attempt = result.training.run_dir / record["paths"]["training"]
        metrics = [
            json.loads(line)["metrics"]
            for line in (attempt / "learner_metrics.jsonl").read_text().splitlines()
        ]
        assert len(metrics) == 2
        assert all(row["train/explained_variance_defined"] == 0 for row in metrics)
        assert all("train/explained_variance" not in row for row in metrics)
    snapshot = read_snapshot(result.training.run_dir)
    assert snapshot["kind"] == snapshot["workflow"]["mode"] == "train-evaluate"
    assert snapshot["workflow"]["title"] == config.logging.title
    assert snapshot["workflow"]["training_unit"] == "adapter decisions"
    assert snapshot["workflow"]["training_seed"] == 101
    assert snapshot["workflow"]["validation_seed"] == 303
    assert snapshot["workflow"]["evaluation_seed"] == 202
    assert snapshot["overview"]["work_total"] == 56
    assert snapshot["overview"]["work_completed"] == 56
    tasks = snapshot["tasks"]
    training = next(row for row in tasks if row["id"] != "evaluation")
    assert training["values"]["validation_batches_finished"] == 2
    assert training["values"]["ppo_updates"] == 2
    assert training["values"]["validation_finished"] == 2
    assert training["values"]["validation_case_active"] is False
    evaluation = next(row for row in tasks if row["id"] == "evaluation")
    assert evaluation["values"]["evaluation_finished"] == 2
    assert evaluation["values"]["evaluation_tick_limit"] == 8
    assert evaluation["values"]["evaluation_case_active"] is False


def test_monitor_rejects_bad_workflow_before_accepting_snapshot(tmp_path):
    result = view(count=1)
    result.bind(tmp_path)
    path = tmp_path / "logs/progress.json"
    saved = json.loads(path.read_text())
    saved["workflow"]["round_size"] = float("nan")
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="workflow"):
        read_snapshot(tmp_path)


@pytest.mark.parametrize("kind", ["training", "train-evaluate", "evaluation", "run"])
def test_shared_live_screen_never_clears_individual_lines(monkeypatch, kind):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = io.StringIO()
    stream.isatty = lambda: True
    console = Console(
        file=stream, force_terminal=True, force_interactive=True, width=100, height=35
    )
    result = RuntimeDisplay(console=console, kind=kind)
    result.configure(metadata(kind))
    result.start()
    result.update(
        "training",
        {"stage": "sampling", "physical_ticks": 128, "updates": 0},
        total=16384,
        unit="physical ticks",
    )
    assert "\x1b[?1049h" in stream.getvalue()
    assert "\x1b[2K" not in stream.getvalue() and "\x1b[2J" not in stream.getvalue()
    result.finish("completed")
    result.close()
    assert "\x1b[?1049l" in stream.getvalue()


def test_batch_header_does_not_label_all_variants_homogeneous(tmp_path):
    from smartsom.telemetry.workflow import describe_study

    scenario = {
        "factory": {
            "factory_id": "factory_007",
            "name": "Small G0 · homogeneous",
            "grid": {"width": 19, "height": 9},
        },
        "tick_limit": 4096,
    }
    (tmp_path / "snapshot.json").write_text(
        json.dumps(
            {"scenario_json": json.dumps(scenario), "config_json": json.dumps({})}
        )
    )
    plan = {
        "recipe": {"training_seed": 101, "total_ticks": 16384},
        "entries": [
            {"id": "h0", "snapshot": "snapshot.json", "H_case": "h0"},
            {"id": "h1", "H_case": "h1"},
        ],
    }
    info = describe_study(tmp_path, plan)
    assert info["factory"] == "factory_007"
    assert info["H_cases"] == ["h0", "h1"] and info["map"] == "19×9"
