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
        == "incomplete"
    )


def test_batch_tracking_group_covers_appended_trials(tmp_path):
    from smartsom.config.base_diagnostic import TrackingOptions
    from smartsom.telemetry.experiment_tracking import ExperimentTracking

    for container in ("experiments", "seeds", "added"):
        tracker = ExperimentTracking(
            tmp_path / "batch123" / container / "trial",
            TrackingOptions(mode="disabled"),
            job_type="evaluate",
        )
        assert tracker.settings["group"] == "batch123"
