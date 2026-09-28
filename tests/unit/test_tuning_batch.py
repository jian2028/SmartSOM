"""Frozen inputs, new-run import and ledger preservation at driver interruption."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import digest
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments import tuning_batch as batch
from smartsom.experiments.composable import archive_inputs
from smartsom.experiments.evidence import write_json

ROOT = Path(__file__).resolve().parents[2]


def prepared():
    return prepare_v3(
        api.load_config(ROOT / "configs/test/runs/train_machine_ppo.yaml"),
        training=True,
    )


def test_calibration_does_not_reuse_unleased_driver_ram():
    from smartsom.experiments.tuning_resources import (
        ProcessUsage,
        ResourceBroker,
        ResourceSnapshot,
    )

    class Monitor:
        def snapshot(self, exclude_pids=()):
            return ResourceSnapshot(
                8,
                16 * 1024**3,
                4 * 1024**3,
                external_cpu_load=10,
                processes=(ProcessUsage(123, "driver", 1, 2 * 1024**3, True),),
            )

    observed = batch.CalibrationMonitor(Monitor()).snapshot((123,))
    capacity = ResourceBroker(mode="throughput").capacity(observed)
    assert observed.external_cpu_load == 10
    assert not observed.processes[0].excluded
    assert capacity.memory < 4 * 1024**3


def test_distinct_training_groups_cannot_share_calibration():
    from dataclasses import replace

    original = prepared()
    data = json.loads(original.config_json)
    data["training"]["groups"] = ["machine", "buffer", "dispatcher"]
    other = replace(original, config_json=json.dumps(data))
    _, mapping = batch._groups(
        [
            {"experiment_id": "one", "prepared": asdict(original)},
            {"experiment_id": "all", "prepared": asdict(other)},
        ]
    )
    assert mapping["one"] != mapping["all"]


def test_manifest_freezes_exact_science_without_starting_learner(tmp_path, monkeypatch):
    import yaml

    original = prepared()
    calls = []

    def prepare(config, **kwargs):
        calls.append(kwargs)
        return original

    monkeypatch.setattr(batch, "prepare_v3", prepare)
    path = tmp_path / "batch.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": batch.SCHEMA,
                "entries": [
                    {
                        "id": "trial",
                        "config": str(
                            ROOT / "configs/test/runs/train_machine_ppo.yaml"
                        ),
                    }
                ],
                "output_root": "results",
            }
        )
    )
    inputs = batch.load_batch(path)
    assert inputs.entries[0]["prepared"] == asdict(original)
    assert calls == [{"training": True, "require_dependencies": True}]
    assert inputs.output_root == str(tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_prepared_study_import_uses_new_identity_and_preserves_source(tmp_path):
    root = tmp_path / "old-study"
    (root / "snapshots/a/config").mkdir(parents=True)
    frozen = archive_inputs(root / "snapshots/a", prepared())
    snapshot = root / "snapshots/a/config/prepared.json"
    plan = {
        "schema": "smartsom.composable-study-plan/v1",
        "source": {"git": "old"},
        "recipe": {"controls": ["initial", "rule", "random"]},
        "entries": [
            {
                "id": "a",
                "snapshot": "snapshots/a/config/prepared.json",
                "snapshot_sha256": digest(json.loads(snapshot.read_text())),
                "scientific_sha256": frozen.scientific_sha256,
            }
        ],
    }
    write_json(root / "plan.json", plan)
    write_json(root / "study.json", {"plan_sha256": digest(plan)})
    before = {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    inputs = batch.load_batch(study=root)
    target, saved, state = batch.allocate_batch(inputs)
    assert target != root and saved["provenance"]["kind"] == "study-input-import"
    assert saved["entries"][0]["prepared"] == asdict(frozen)
    assert state["entries"]["a"]["status"] == "queued"
    assert {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == before
    changed = json.loads(snapshot.read_text())
    changed["scientific_sha256"] = "changed"
    write_json(snapshot, changed)
    with pytest.raises(ValueError, match="snapshot"):
        batch.load_batch(study=root)


def test_run_rejects_plan_drift_before_calibration(tmp_path):
    inputs = batch.BatchInputs(
        ({"experiment_id": "a", "prepared": asdict(prepared()), "control_spec": {}},),
        output_root=str(tmp_path),
    )
    root, plan, _ = batch.allocate_batch(inputs)
    plan["execution"] = "fixed"
    write_json(root / "plan.json", plan)
    with pytest.raises(ValueError, match="plan changed"):
        batch.load_run(root)


def test_interrupt_never_overwrites_newer_verified_ledger(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    inputs = batch.BatchInputs(
        ({"experiment_id": "a", "prepared": asdict(prepared()), "control_spec": {}},),
        output_root=str(tmp_path),
    )
    root, plan, stale = batch.allocate_batch(inputs)

    def interrupt(root, plan, **kwargs):
        actual = json.loads((root / "batch.json").read_text())
        actual["entries"]["a"].update(
            status="completed", commit_id="verified-complete", checkpoint="durable"
        )
        write_json(root / "batch.json", actual)
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "calibrate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        batch.execute_batch(root, plan, stale)
    actual = json.loads((root / "batch.json").read_text())
    assert actual["status"] == "interrupted"
    assert actual["entries"]["a"]["commit_id"] == "verified-complete"
    assert actual["entries"]["a"]["status"] == "completed"


def test_completed_batch_is_not_calibrated_or_trained_again(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    inputs = batch.BatchInputs(
        ({"experiment_id": "a", "prepared": asdict(prepared()), "control_spec": {}},),
        output_root=str(tmp_path),
    )
    root, plan, state = batch.allocate_batch(inputs)
    state["entries"]["a"]["status"] = "completed"

    def forbidden(*args, **kwargs):
        raise AssertionError("completed experiments must be skipped")

    monkeypatch.setattr(batch, "calibrate", forbidden)
    assert batch.execute_batch(root, plan, state)["completed"] == 1
    assert json.loads((root / "batch.json").read_text())["status"] == "completed"


def test_framework_diagnostics_do_not_pollute_json_stdout(capsys):
    batch._framework_call(print, "Ray human readable header")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Ray human readable header" in captured.err


def test_resume_calibrates_only_eligible_unfinished_entries(tmp_path, monkeypatch):
    entries = tuple(
        {"experiment_id": name, "prepared": asdict(prepared()), "control_spec": {}}
        for name in ("complete", "failed", "queued")
    )
    root, plan, state = batch.allocate_batch(
        batch.BatchInputs(entries, output_root=str(tmp_path))
    )
    state["entries"]["complete"]["status"] = "completed"
    state["entries"]["failed"]["status"] = "failed"

    def stop(root, plan, **kwargs):
        assert [e["experiment_id"] for e in plan["entries"]] == ["queued"]
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "calibrate", stop)
    with pytest.raises(KeyboardInterrupt):
        batch.execute_batch(root, plan, state)


def test_fixed_resume_reuses_valid_calibration(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")
    inputs = batch.BatchInputs(
        ({"experiment_id": "a", "prepared": asdict(prepared()), "control_spec": {}},),
        execution="fixed",
        output_root=str(tmp_path),
    )
    root, plan, state = batch.allocate_batch(inputs)
    state["entries"]["a"]["attempts"] = [
        {"run_dir": str(root / "experiments/a/attempt-0001")}
    ]
    write_json(root / "calibration.json", {"ready": True})
    monkeypatch.setattr(
        batch,
        "calibrate",
        lambda *args, **kwargs: pytest.fail("fixed resume recalibrated"),
    )
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="fresh local Ray"):
        batch.execute_batch(root, plan, state)


def test_waiting_profile_is_a_serializable_display_fact(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from smartsom.experiments import tuning_calibration
    from smartsom.experiments.tuning_resources import ExecutionProfile, ResourceSnapshot
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    class Monitor:
        def snapshot(self, exclude_pids=()):
            return ResourceSnapshot(8, 16 * 1024**3, 12 * 1024**3, external_cpu_load=0)

    class Controller:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, groups, candidates, **kwargs):
            kwargs["on_wait"](
                {"profile": ExecutionProfile(1, 1), "waiting_seconds": 12}
            )
            assert view.tuning["calibration"]["profile"] == {
                "threads": 1,
                "concurrency": 1,
                "device": "cpu",
            }
            assert view.tuning["calibration"]["waiting_seconds"] == 12
            return SimpleNamespace(
                active_seconds=0,
                waiting_seconds=12,
                measurements=(),
                recommendations={},
                status="completed",
                reason=None,
                missing_groups=(),
                converged_groups=(),
                unstable_groups=(),
                ready=True,
            )

    monkeypatch.setattr(tuning_calibration, "CalibrationController", Controller)
    view = RuntimeDisplay(
        DisplayOptions(progress="off", verbose=False), kind="tune", quiet=True
    )
    plan = {
        "entries": [{"experiment_id": "a", "prepared": asdict(prepared())}],
        "mode": "office",
        "active_limit": 30,
    }
    batch.calibrate(
        tmp_path, plan, display=view, monitor=Monitor(), supervisor=object()
    )
