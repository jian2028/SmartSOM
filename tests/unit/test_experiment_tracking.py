"""Failure, identity and RNG boundaries without making cloud requests."""

import json
import random
import sys
from types import SimpleNamespace

import pytest

from smartsom.config.base_diagnostic import TrackingOptions
from smartsom.telemetry.experiment_tracking import ExperimentTracking, resolved


def manifest():
    return {"source": {"commit": "abc", "implementation_sha256": "def"}, "inputs": {}}


@pytest.mark.parametrize(
    "name,method,observation",
    [
        ("ppo", "ppo", "base"),
        ("sl-dqn", "dqn", "sl"),
        ("rule", "rule", "base"),
        ("random", "random", "base"),
    ],
)
def test_frozen_tracking_labels(tmp_path, name, method, observation):
    from pathlib import Path
    from shutil import copyfile

    from smartsom.telemetry.tracking_metadata import metadata

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    configs = Path(__file__).resolve().parents[2] / "configs"
    copyfile(
        configs
        / "algorithms"
        / (("base-" if name in {"ppo", "dqn"} else "") + name + ".yaml"),
        inputs / "algorithm.yaml",
    )
    copyfile(configs / "factories/small.yaml", inputs / "factory.yaml")
    result = metadata(
        tmp_path,
        {"run": {"output": "runs/example"}},
        seed=4101,
        tags=("smoke",),
        variant="explicit",
    )
    assert result["comparison"]["method"] == method
    assert result["comparison"]["observation"] == observation
    assert result["name"].startswith(f"example－small－{method}－explicit")
    assert result["name"].endswith("s4101") == (method in {"ppo", "dqn"})
    assert "map:small" in result["tags"] and "smoke" in result["tags"]
    assert (
        result["documents"]["algorithm"]["schema"] == "smartsom.algorithm-components/v1"
    )


class Run:
    url, entity, project = (
        "https://wandb.ai/example/SmartSOM/runs/test",
        "example",
        "SmartSOM",
    )
    sync_dir = "/tmp/wandb-test"

    def __init__(self):
        self.events = []

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values):
        random.random()
        self.events.append(values)

    def finish(self, **kwargs):
        pass


class Table:
    def __init__(self, *, columns, log_mode):
        self.columns, self.log_mode, self.data = columns, log_mode, []

    def add_data(self, *row):
        self.data.append(list(row))


def sdk(monkeypatch, *, fail_init=False):
    calls, runs = [], []

    def init(**kwargs):
        calls.append(kwargs)
        random.random()
        if fail_init and kwargs["mode"] == "online":
            raise ConnectionError("offline")
        run = Run()
        runs.append(run)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Settings=lambda **kw: kw,
            Table=Table,
        ),
    )
    return calls, runs


def test_default_online_and_explicit_environment_modes(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert resolved(TrackingOptions())["mode"] == "online"
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert resolved(TrackingOptions())["mode"] == "offline"
    assert resolved(TrackingOptions(mode="disabled"))["mode"] == "disabled"


def test_online_resume_and_rng_are_independent(tmp_path, monkeypatch):
    calls, runs = sdk(monkeypatch)
    before = random.getstate()
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    )
    tracker.start(manifest(), {})
    tracker.log({"physical_ticks": 100, "losses": {"machine": {"q": 0.1}}})
    tracker.close()
    assert random.getstate() == before
    resumed = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    )
    resumed.start(manifest(), {})
    resumed.log({"physical_ticks": 200})
    resumed.close()
    assert calls[0]["id"] == calls[1]["id"] and calls[1]["resume"] == "must"
    assert calls[0]["settings"]["save_code"] is False
    assert runs[1].events[0]["event_sequence"] == 2
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["tracking"]["entity"]
        == "example"
    )


def test_offline_resume_has_distinct_segments(tmp_path, monkeypatch):
    calls, _ = sdk(monkeypatch)
    identities = []
    for _ in range(2):
        tracker = ExperimentTracking(
            tmp_path, TrackingOptions(mode="offline"), job_type="train"
        )
        tracker.start(manifest(), {})
        identities.append(tracker.status["experiment_id"])
        tracker.close()
    assert identities[0] == identities[1]
    assert calls[0]["id"] != calls[1]["id"]
    assert all(c["resume"] is None for c in calls)


def test_noninteractive_startup_failure_is_not_silently_offline(tmp_path, monkeypatch):
    calls, _ = sdk(monkeypatch, fail_init=True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    )
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="Explicitly"):
        tracker.start(manifest(), {})
    assert len(calls) == 1
    assert tracker.stream is None


def test_interactive_failure_requires_explicit_offline_choice(tmp_path, monkeypatch):
    calls, _ = sdk(monkeypatch, fail_init=True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    )
    with pytest.warns(UserWarning):
        tracker.start(manifest(), {})
    tracker.close()
    assert [c["mode"] for c in calls] == ["online", "offline"]


def test_upload_and_finish_failure_preserve_local_record_and_checkpoint(
    tmp_path, monkeypatch
):
    _, runs = sdk(monkeypatch)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"durable-checkpoint")
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    )
    tracker.start(manifest(), {})

    def fail(*args, **kwargs):
        raise TimeoutError("network unavailable")

    runs[0].log = runs[0].finish = fail
    with pytest.warns(UserWarning):
        tracker.log({"physical_ticks": 42})
    with pytest.warns(UserWarning):
        tracker.close()
    assert checkpoint.read_bytes() == b"durable-checkpoint"
    rows = (tmp_path / "tracking/events.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["physical_ticks"] == 42
    assert (
        json.loads((tmp_path / "tracking/status.json").read_text())["sync_status"]
        == "upload_error"
    )
    assert tracker.status["finish_error"] == "TimeoutError"


@pytest.mark.parametrize("monitor_error", [False, True])
def test_rejected_event_remains_unsynced_after_poll_finish_and_resume(
    tmp_path, monkeypatch, monitor_error
):
    import threading

    _, runs = sdk(monkeypatch)
    # Exercise one poll synchronously without a sleeping monitor thread.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    ).start(manifest(), {})

    def reject(*args, **kwargs):
        raise ConnectionError("rejected before enqueue")

    monkeypatch.setattr(runs[0], "log", reject)
    with pytest.warns(UserWarning, match="upload failed"):
        tracker.log({"physical_ticks": 1})
    assert not runs[0].events
    polls = iter((False, True))
    monkeypatch.setattr(tracker.stop, "wait", lambda _: next(polls))
    runs[0].status = (
        reject
        if monitor_error
        else lambda: SimpleNamespace(sync_time=1, sync_items_pending=0)
    )
    tracker._monitor()
    assert tracker.status["sync_status"] == "upload_error"
    monkeypatch.setattr(runs[0], "log", Run.log.__get__(runs[0]))
    tracker.log({"physical_ticks": 2})
    tracker.close()
    assert len(runs[0].events) == 1
    state = json.loads((tmp_path / "tracking/status.json").read_text())
    assert state["sync_status"] == "upload_error"
    assert state["upload_errors"] == 1 and state["last_failed_event_sequence"] == 1
    assert len((tmp_path / "tracking/events.jsonl").read_text().splitlines()) == 2
    resumed = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    ).start(manifest(), {})
    resumed.close()
    assert resumed.status["sync_status"] == "upload_error"
    assert resumed.status["last_upload_error"] == "ConnectionError"


def test_batch_tracking_group_covers_appended_trials(tmp_path):
    for container in ("experiments", "seeds", "added"):
        root = tmp_path / "batch123" / container / "trial"
        root.mkdir(parents=True)
        tracker = ExperimentTracking(
            root, TrackingOptions(mode="disabled"), job_type="evaluate"
        )
        tracker.start(
            {**manifest(), "tracking_group": "batch123-20261030T000000Z-abcdef"}, {}
        )
        assert tracker.settings["group"] == "batch123-20261030T000000Z-abcdef"
        tracker.close()


def test_throttle_keeps_local_events_and_flushes_latest(tmp_path, monkeypatch):
    sdk(monkeypatch)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="offline"), job_type="train"
    )
    tracker.start(manifest(), {})
    tracker.log({"phase": "train", "physical_ticks": 1})
    tracker.log(
        {
            "phase": "train",
            "physical_ticks": 2,
            "final_station_state": {"secret_entity": {"count": 999}},
        }
    )
    assert len(tracker.run.events) == 1
    assert len((tmp_path / "tracking/events.jsonl").read_text().splitlines()) == 2
    tracker.close()
    assert len(tracker.run.events) == 2
    assert tracker.run.events[-1]["train/physical_ticks"] == 2
    assert not any("secret_entity" in key for key in tracker.run.events[-1])


def test_resume_keeps_original_group_and_summary_is_immediate(tmp_path, monkeypatch):
    sdk(monkeypatch)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="offline"), job_type="train"
    )
    tracker.start(manifest(), {})
    original = tracker.settings["group"]
    tracker.log(
        {
            "phase": "validation",
            "selection": "sample",
            "kind": "summary",
            "training_progress": {"episodes": 3},
            "methods": {
                "ppo": {
                    "all": {
                        "metrics": {"joint_cost": 2.5},
                        "actual_cases": 4,
                        "planned_cases": 4,
                        "complete_inventory": True,
                    }
                }
            },
        }
    )
    assert tracker.run.events[-1]["validation/ppo/sample/all/joint_cost"] == 2.5
    tracker.close()
    again = ExperimentTracking(
        tmp_path, TrackingOptions(mode="offline"), job_type="train"
    )
    again.start(manifest(), {})
    assert again.settings["group"] == original
    again.close()


def test_medium_metadata_uses_real_factory_identity(tmp_path):
    from pathlib import Path
    from shutil import copyfile

    from smartsom.telemetry.tracking_metadata import metadata

    source = Path(__file__).resolve().parents[2] / "configs"
    (tmp_path / "inputs").mkdir()
    copyfile(source / "factories/medium.yaml", tmp_path / "inputs/factory.yaml")
    assert metadata(tmp_path, {})["comparison"]["map"] == "medium"


def test_summary_tables_append_only_new_rows(tmp_path, monkeypatch):
    sdk(monkeypatch)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="offline"), job_type="train"
    )
    tracker.start({**manifest(), "seed": 101}, {"train_seeds": [101, 102]})
    for episode in (1, 2):
        tracker.log(
            {
                "phase": "validation",
                "selection": "sample",
                "kind": "summary",
                "training_progress": {"episodes": episode},
                "methods": {
                    "ppo": {
                        "all": {
                            "metrics": {"joint_cost": float(episode)},
                            "actual_cases": 4,
                            "planned_cases": 4,
                            "complete_inventory": True,
                        }
                    }
                },
            }
        )
    table = tracker.tables["research/summary_history"]
    assert table.log_mode == "INCREMENTAL" and len(table.data) == 2
    assert [row[table.columns.index("episode")] for row in table.data] == [1, 2]
    tracker.close()


@pytest.mark.parametrize("level", ["summary", "detailed"])
def test_real_buffer_entities_and_each_method_representative(tmp_path, level):
    from smartsom.config.base_diagnostic import BaseWorkload, TrackingOptions
    from smartsom.config.base_layout import base_factory
    from smartsom.experiments.reservation_runtime import episode
    from smartsom.policies.rule import BaseRule
    from smartsom.telemetry.experiment_tracking import (
        ExperimentTracking,
        representative_case,
    )
    from tests.helpers.algorithm import algorithm_options

    workload = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=4,
        initial_orders=2,
        mean_interarrival=8,
    )
    options = algorithm_options(schema="smartsom.base-algorithm/v1", method="rule")
    result, _, _ = episode(
        base_factory("small"),
        workload,
        options,
        BaseRule("rule", 51),
        71,
        "normal",
        30,
        max_ticks=20,
        metrics_level=level,
    )
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="disabled"), job_type="evaluate"
    )
    tracker.start({"source": {}, "inputs": {}}, {})
    try:
        event = {
            **result,
            "phase": "evaluation",
            "method": "random",
            "selection": "sample",
            "action_seed": 0,
            "case_index": 4,
            "evaluation_identity": {"world_index": 0},
        }
        tracker.log(event)
        expected = result["research_metrics"]["buffer_entities"]
        actual = {
            r["entity"]: r
            for r in tracker.entity_rows.values()
            if r["kind"] == "buffers"
        }
        assert actual.keys() == expected.keys()
        assert len(actual) == 18
        for entity, counts in expected.items():
            assert actual[entity]["resident_ticks"] == counts["resident_ticks"]
            assert actual[entity]["observed_ticks"] == counts["observed_ticks"]
        assert representative_case(event)
        assert not representative_case({**event, "condition": "high"})
        assert not representative_case(
            {**event, "evaluation_identity": {"world_index": 1}}
        )
        assert tracker.status["entity_series_cases"]["sample"]["random"]["case"] == 4
    finally:
        tracker.close()


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_sync_command_replays_the_recorded_segment(monkeypatch, tmp_path, mode):
    import shlex

    sdk(monkeypatch)
    monkeypatch.setattr(Run, "sync_dir", "/tmp/run with 'quote' and $(literal)")
    tracker = ExperimentTracking(tmp_path, TrackingOptions(mode=mode), job_type="train")
    tracker.start(manifest(), {})
    try:
        command = shlex.split(tracker.status["sync_command"])
        assert command[:2] == ["wandb", "sync"]
        assert ("--no-skip-online" in command) == (mode == "online")
        assert command[-1] == Run.sync_dir
    finally:
        tracker.close()


def test_foreground_worker_relays_offline_consent_to_terminal(tmp_path, monkeypatch):
    from smartsom.experiments import foreground

    calls, _ = sdk(monkeypatch, fail_init=True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv(foreground.PROMPT_ENV, str(tmp_path))
    requests = []

    def choice(error):
        requests.append(error)
        return True

    monkeypatch.setattr(foreground, "offline_choice", choice)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="online"), job_type="train"
    )
    with pytest.warns(UserWarning):
        tracker.start(manifest(), {})
    tracker.close()
    assert len(requests) == 1
    assert [call["mode"] for call in calls] == ["online", "offline"]
