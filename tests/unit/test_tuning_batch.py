"""Frozen inputs, new-run import and ledger preservation at driver interruption."""

import json
import time
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


def test_frozen_runtime_identity_allows_unrelated_dirty_status_only():
    from smartsom.experiments.evidence import runtime_source_matches

    frozen = {
        "python": "3.12",
        "packages": {"ray": "2.58.0"},
        "git": {"commit": "abc", "status": " M README.md"},
    }
    live = {**frozen, "git": {"commit": "abc", "status": " M docs/new.md"}}
    assert runtime_source_matches(frozen, live)
    assert not runtime_source_matches(frozen, {**live, "python": "3.13"})
    assert not runtime_source_matches(
        frozen, {**live, "git": {"commit": "def", "status": ""}}
    )


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
    capacity = ResourceBroker(mode="performance").capacity(observed)
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


def test_pickup_matching_changes_calibration_group():
    from dataclasses import replace

    original = prepared()
    composition = json.loads(original.composition_json)
    composition["matching"] = {
        "schema": "smartsom.pickup-matching/v2",
        "name": "priority_greedy",
    }
    other = replace(original, composition_json=json.dumps(composition))
    _, mapping = batch._groups(
        [
            {"experiment_id": "optimal", "prepared": asdict(original)},
            {"experiment_id": "greedy", "prepared": asdict(other)},
        ]
    )
    assert mapping["optimal"] != mapping["greedy"]


@pytest.mark.parametrize("kind", ["author-batch", "author-plan"])
def test_directory_batch_reuses_one_bound_calibration_before_training(
    tmp_path, monkeypatch, kind
):
    from smartsom.experiments.tuning_resources import ExecutionProfile

    original = prepared()
    entries = (
        {"experiment_id": "one", "prepared": asdict(original), "control_spec": {}},
    )
    inputs = batch.BatchInputs(
        entries,
        output_root=str(tmp_path),
        provenance={"kind": kind, "parent_plan_sha256": "test"},
    )
    root, plan, state = batch.allocate_batch(inputs)
    group = batch._groups(plan["entries"])[1]["one"]
    calls = []

    def measure(*args, **kwargs):
        calls.append(1)
        return {
            "ready": True,
            "groups": {"one": group},
            "recommendations": {group: asdict(ExecutionProfile(1, 1, "cpu"))},
            "reason": "measured",
            "active_seconds": 1.0,
            "waiting_seconds": 0.0,
            "wall_seconds": 1.0,
            "calibrated": True,
            "uncalibrated_groups": [],
        }

    monkeypatch.setattr(batch, "calibrate", measure)
    first = batch._execute_batch(root, plan, state, recommend_only=True)
    assert first["status"] == "recommended"
    second = batch._execute_batch(root, plan, state, recommend_only=True)
    assert second["status"] == "recommended"
    assert len(calls) == 1


def test_explicit_off_records_fixed_uncalibrated_layout_without_probes(
    tmp_path, monkeypatch
):
    original = prepared()
    entries = tuple(
        {
            "experiment_id": identity,
            "prepared": asdict(original),
            "control_spec": {},
            "baseline_concurrency": 2,
        }
        for identity in ("one", "two")
    )
    inputs = batch.BatchInputs(
        entries,
        mode="performance",
        execution="fixed",
        active_limit=0.0,
        calibration_level="off",
        output_root=str(tmp_path),
        provenance={"kind": "author-batch", "parent_plan_sha256": "test"},
    )
    root, plan, _ = batch.allocate_batch(inputs)
    monkeypatch.setattr(
        batch,
        "CalibrationMonitor",
        lambda *_args, **_kwargs: pytest.fail("resource probing must be skipped"),
    )
    result = batch.calibrate(root, plan)
    assert result["status"] == "skipped"
    assert result["calibrated"] is False
    assert result["measurements"] == []
    assert result["active_seconds"] == 0
    assert result["recommendations"]
    assert all(
        profile["concurrency"] == 2 for profile in result["recommendations"].values()
    )
    assert set(result["uncalibrated_groups"]) == set(result["recommendations"])
    assert (root / "calibration.json").is_file()


@pytest.mark.parametrize(
    ("second_seconds", "second_ticks", "expected"),
    [(10.0, 1.0, "separate_measured"), (1.0, 100.0, "mixed_measured")],
)
def test_mixed_probe_ranks_training_makespan_not_aggregate_throughput(
    tmp_path, second_seconds, second_ticks, expected
):
    from smartsom.experiments.tuning_calibration import CandidateMeasurement
    from smartsom.experiments.tuning_resources import (
        ExecutionProfile,
        ResourceSnapshot,
    )

    original = asdict(prepared())
    entries = [{"experiment_id": name, "prepared": original} for name in ("a", "b")]
    profile = ExecutionProfile(1, 1, "cpu")
    measured = [
        CandidateMeasurement(
            profile,
            throughput=10.0,
            peak_memory=100,
            stages={
                "per_trial_peak_memory": 100.0,
                "worker_0_seconds": 1.0,
                "sampling": 0.4,
                "saving": 0.1,
                "validation": 0.5,
            },
            elapsed_seconds=1.0,
            group=name,
        )
        for name in ("A", "B")
    ]

    class Monitor:
        def snapshot(self, exclude_pids=()):
            return ResourceSnapshot(8, 16 * 1024**3, 8 * 1024**3)

    class Supervisor:
        def run(self, probe, group, selected, remaining, cancelled):
            assert len(group["mixed_workers"]) == 2
            assert selected.concurrency == 2
            return CandidateMeasurement(
                selected,
                throughput=(100 + second_ticks) / 10,
                peak_memory=200,
                stages={
                    "worker_0_seconds": 1.0,
                    "worker_0_ticks": 100.0,
                    "worker_0_sampling": 0.2,
                    "worker_0_saving": 0.1,
                    "worker_0_validation": 0.1,
                    "worker_1_seconds": second_seconds,
                    "worker_1_ticks": second_ticks,
                    "worker_1_sampling": 8.0 if second_seconds > 1 else 0.2,
                    "worker_1_saving": 0.1,
                    "worker_1_validation": 1.0 if second_seconds > 1 else 0.1,
                },
                elapsed_seconds=10.0,
            )

    plan = {"entries": entries, "mode": "performance", "active_limit": 1000}
    schedule = batch._batch_schedule(
        tmp_path,
        plan,
        {name: {"prepared": original} for name in ("A", "B")},
        {"a": "A", "b": "B"},
        {name: profile for name in ("A", "B")},
        measured,
        Supervisor(),
        Monitor(),
        time.monotonic(),
        cancelled=lambda: False,
    )
    assert schedule["status"] == expected
    assert (len(schedule["waves"]) == 1) == (expected == "mixed_measured")


def test_mixed_probe_does_not_extrapolate_two_workers_to_four(tmp_path):
    from smartsom.experiments.tuning_calibration import CandidateMeasurement
    from smartsom.experiments.tuning_resources import ExecutionProfile, ResourceSnapshot

    original = asdict(prepared())
    profile = ExecutionProfile(1, 2, "cpu")
    entries = [{"experiment_id": name, "prepared": original} for name in ("a", "b")]
    measurements = [
        CandidateMeasurement(profile, 20.0, 200, elapsed_seconds=1.0, group=name)
        for name in ("A", "B")
    ]

    class Monitor:
        def snapshot(self, exclude_pids=()):
            return ResourceSnapshot(8, 16 * 1024**3, 8 * 1024**3)

    class Supervisor:
        def run(self, *args):
            pytest.fail("a two-worker pilot cannot validate four-worker execution")

    schedule = batch._batch_schedule(
        tmp_path,
        {"entries": entries, "mode": "performance", "active_limit": 1000},
        {name: {"prepared": original} for name in ("A", "B")},
        {"a": "A", "b": "B"},
        {name: profile for name in ("A", "B")},
        measurements,
        Supervisor(),
        Monitor(),
        time.monotonic(),
        cancelled=lambda: False,
    )
    assert schedule["status"] == "schedule_uncalibrated"
    assert len(schedule["waves"]) == 2


def test_short_probe_refuses_longer_update_or_validation_case():
    from smartsom.experiments.tuning_batch import _projection_compatible

    full = asdict(prepared())
    probe = dict(full)
    full_config = json.loads(full["config_json"])
    probe_config = json.loads(probe["config_json"])
    full_config["training"].update(total_ticks=2048, ticks_per_update=1024)
    probe_config["training"].update(total_ticks=512, ticks_per_update=512)
    full["config_json"] = json.dumps(full_config)
    probe["config_json"] = json.dumps(probe_config)
    assert not _projection_compatible([{"prepared": full}], {"prepared": probe})
    full_config["training"]["ticks_per_update"] = 512
    full["config_json"] = json.dumps(full_config)
    cases = json.loads(full["validation_json"])
    cases[0]["scenario"]["tick_limit"] = 4096
    full["validation_json"] = json.dumps(cases)
    assert not _projection_compatible([{"prepared": full}], {"prepared": probe})


def test_selected_sampling_layout_changes_new_training_identity():
    original = prepared()
    selected = batch._with_selected_layout(
        original,
        {"threads": 2, "num_envs": 4, "sampling_processes": 2},
    )
    assert selected.config.runtime.num_envs == 4
    assert selected.config.runtime.sampling_processes == 2
    assert selected.scientific_sha256 != original.scientific_sha256
    assert original.config.runtime.num_envs == 1


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
                "num_envs": 1,
                "sampling_processes": 0,
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
        "mode": "balanced",
        "active_limit": 30,
    }
    batch.calibrate(
        tmp_path, plan, display=view, monitor=Monitor(), supervisor=object()
    )


def test_sampling_preflight_never_constructs_learning_policies(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("smartsom.learning.production_inference"):
            raise ImportError("learning frameworks unavailable in base environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    assert batch._parallel_sampling_issue(prepared()) is None


@pytest.mark.parametrize("kind", ("new_model", "model"))
def test_sampling_preflight_checks_observation_registration_without_construction(kind):
    from dataclasses import replace

    from smartsom.learning.extensions import register_extension

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("preflight must not construct observation encoders")

    entry = register_extension(
        "observation",
        "test.tuning_preflight_" + kind,
        "1",
        forbidden_factory,
        supported_backends=("rllib.resource_ppo",),
        stateful=True,
    )
    original = prepared()
    declarations = json.loads(original.policies_json)
    reference = {
        "name": entry.name,
        "version": entry.version,
        "code_sha256": entry.code_sha256,
    }
    if kind == "model":
        declarations["machine"]["implementation"] = {"kind": "model"}
        declarations["machine"]["resolved_model"] = {
            "metadata": {"observation": reference, "provider": "rllib.resource_ppo"}
        }
    else:
        declarations["machine"]["implementation"]["extensions"]["observation"] = (
            reference
        )
    frozen = replace(original, policies_json=json.dumps(declarations))
    assert batch._parallel_sampling_issue(frozen) == (
        "parallel sampling cannot merge stateful observation group machine"
    )
    reference["code_sha256"] = "0" * 64
    changed = replace(original, policies_json=json.dumps(declarations))
    with pytest.raises(ValueError, match="digest changed"):
        batch._parallel_sampling_issue(changed)


def test_sampling_preflight_checks_rule_registration_without_construction():
    from dataclasses import replace

    from smartsom.algorithms.rule_registry import register_rule

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("preflight must not construct rules")

    entry = register_rule(
        "test.tuning_rule_preflight",
        "1",
        forbidden_factory,
        roles=("machine",),
        stateful=True,
    )
    original = prepared()
    declarations = json.loads(original.policies_json)
    declarations["machine"]["implementation"] = {
        "kind": "rule",
        "name": entry.name,
        "version": entry.version,
        "parameters": {},
        "code_sha256": entry.code_sha256,
    }
    frozen = replace(original, policies_json=json.dumps(declarations))
    assert batch._parallel_sampling_issue(frozen) == (
        "parallel sampling cannot merge stateful rule group machine"
    )
