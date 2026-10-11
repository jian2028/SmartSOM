"""Frozen source and unsafe process ownership fail visibly instead of waiting."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from smartsom.experiments.tuning_callbacks import EvidenceCallback


def test_online_feedback_uses_verified_commit_counts_and_rejects_invalid_commit(
    tmp_path, monkeypatch
):
    from dataclasses import asdict
    from pathlib import Path

    from smartsom import api
    from smartsom.config.experiment_v3 import prepare_v3
    from smartsom.experiments import tuning_callbacks

    root = Path(__file__).resolve().parents[2]
    prepared = prepare_v3(
        api.load_config(root / "configs/test/runs/train_machine_ppo.yaml"),
        training=True,
    )
    (tmp_path / "batch.json").write_text(
        json.dumps({"entries": {"a": {"status": "queued"}}})
    )
    observed = []
    broker = SimpleNamespace(
        note_actor=lambda *args: None,
        observe_formal_update=lambda identity, result: observed.append(
            (identity, result)
        ),
    )
    trial = SimpleNamespace(
        config={"experiment_id": "a", "prepared": asdict(prepared), "record": {}}
    )
    marker = {
        "phase": "training",
        "updates": 2,
        "physical_ticks": 128,
        "commit_id": "verified",
    }
    monkeypatch.setattr(tuning_callbacks, "verify_identity", lambda *args: marker)
    callback = EvidenceCallback(tmp_path, [], broker)
    result = {
        "checkpoint": "commit",
        "allocation_epoch": 0,
        "updates": 999,
        "physical_ticks": 99999,
    }
    callback.on_trial_result(1, [], trial, result)
    assert observed[0][1]["updates"] == 2 and observed[0][1]["physical_ticks"] == 128

    def invalid(*args):
        raise ValueError("invalid commit")

    monkeypatch.setattr(tuning_callbacks, "verify_identity", invalid)
    with pytest.raises(ValueError, match="invalid commit"):
        callback.on_trial_result(2, [], trial, result)
    assert len(observed) == 1


def _frozen_plan(tmp_path, monkeypatch, *, live_source=None, live_code="frozen"):
    from smartsom.experiments import composable, tuning_callbacks

    source = {"git": {"commit": "start"}}
    (tmp_path / "plan.json").write_text(
        json.dumps({"source": source, "implementation_sha256": "frozen"})
    )
    monkeypatch.setattr(composable, "implementation_identity", lambda: live_code)
    monkeypatch.setattr(
        tuning_callbacks, "source_identity", lambda: live_source or source
    )


def test_incomplete_failed_actor_ownership_stops_segment(tmp_path, monkeypatch):
    _frozen_plan(tmp_path, monkeypatch)
    (tmp_path / "batch.json").write_text(
        json.dumps(
            {
                "entries": {
                    "failed-sampler": {"failure": {"message": "worker setup failed"}}
                }
            }
        )
    )

    class Broker:
        def refresh(self):
            pass

        def unresolved_failures(self):
            return ("failed-sampler",)

    callback = EvidenceCallback(tmp_path, [], Broker())
    with pytest.raises(
        RuntimeError,
        match=r"reservation retained, batch stopped: failed-sampler \(worker setup failed\)",
    ):
        callback.on_step_begin(1, [])


@pytest.mark.parametrize(
    ("live_source", "live_code", "message"),
    [
        (None, "changed", "implementation changed after freezing"),
        ({"git": {"commit": "later"}}, "frozen", "checkout or dependencies changed"),
    ],
)
def test_source_change_aborts_before_more_trials(
    tmp_path, monkeypatch, live_source, live_code, message
):
    _frozen_plan(tmp_path, monkeypatch, live_source=live_source, live_code=live_code)

    class Broker:
        def refresh(self):
            raise AssertionError("changed source must stop before resource admission")

    callback = EvidenceCallback(tmp_path, [], Broker())
    with pytest.raises(RuntimeError, match=message):
        callback.on_step_begin(1, [])


def test_failed_trial_saves_worker_error_for_monitor(tmp_path):
    (tmp_path / "batch.json").write_text(
        json.dumps({"entries": {"entry": {"status": "queued"}}})
    )

    class Broker:
        def actor_failed(self, identity, trial):
            assert identity == "entry"

        def unresolved_failures(self):
            return ()

    trial = SimpleNamespace(
        config={"experiment_id": "entry", "run_dir": str(tmp_path / "attempt")},
        get_pickled_error=lambda: ValueError("worker implementation differs"),
    )
    EvidenceCallback(tmp_path, [], Broker()).on_trial_error(1, [], trial)
    row = json.loads((tmp_path / "batch.json").read_text())["entries"]["entry"]
    assert row["status"] == "failed"
    assert row["failure"] == {
        "exception": "ValueError",
        "message": "worker implementation differs",
    }


def test_worker_sidecar_survives_missing_ray_error(tmp_path):
    run = tmp_path / "attempt"
    (run / "logs").mkdir(parents=True)
    (run / "logs/worker-error.json").write_text(
        json.dumps(
            {
                "exception": "ValueError",
                "message": "worker implementation differs from frozen batch",
                "phase": "setup",
            }
        )
    )
    (tmp_path / "batch.json").write_text(
        json.dumps({"entries": {"entry": {"status": "queued"}}})
    )

    class Broker:
        def actor_failed(self, identity, trial):
            pass

        def unresolved_failures(self):
            return ()

    trial = SimpleNamespace(
        config={"experiment_id": "entry", "run_dir": str(run)},
        get_pickled_error=lambda: None,
    )
    EvidenceCallback(tmp_path, [], Broker()).on_trial_error(1, [], trial)
    row = json.loads((tmp_path / "batch.json").read_text())["entries"]["entry"]
    assert row["failure"] == {
        "exception": "ValueError",
        "message": "worker implementation differs from frozen batch",
    }


def test_missing_ray_error_still_saves_actionable_fallback(tmp_path):
    (tmp_path / "batch.json").write_text(
        json.dumps({"entries": {"entry": {"status": "queued"}}})
    )

    class Broker:
        def actor_failed(self, identity, trial):
            pass

        def unresolved_failures(self):
            return ()

    trial = SimpleNamespace(
        config={"experiment_id": "entry", "run_dir": str(tmp_path / "attempt")},
        get_pickled_error=lambda: None,
    )
    EvidenceCallback(tmp_path, [], Broker()).on_trial_error(1, [], trial)
    row = json.loads((tmp_path / "batch.json").read_text())["entries"]["entry"]
    assert row["failure"]["exception"] == "RayTrialError"
    assert "see trial logs" in row["failure"]["message"]


def test_unknown_failed_actor_aborts_in_error_callback_without_another_step(tmp_path):
    (tmp_path / "batch.json").write_text(
        json.dumps({"entries": {"entry": {"status": "queued"}}})
    )

    class Broker:
        def actor_failed(self, identity, trial):
            assert identity == "entry"

        def unresolved_failures(self):
            return ("entry",)

    trial = SimpleNamespace(
        config={"experiment_id": "entry", "run_dir": str(tmp_path / "attempt")},
        get_pickled_error=lambda: ValueError("worker implementation differs"),
    )
    callback = EvidenceCallback(tmp_path, [], Broker())
    with pytest.raises(
        RuntimeError, match="batch stopped: entry \\(worker implementation differs\\)"
    ):
        callback.on_trial_error(1, [], trial)
    row = json.loads((tmp_path / "batch.json").read_text())["entries"]["entry"]
    assert row["status"] == "failed"
