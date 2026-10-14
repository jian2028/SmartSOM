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


def test_online_model_partners_group_by_fingerprint_instead_of_archive_paths():
    import copy

    original = asdict(prepared())
    entries = []
    for identity in ("a", "b"):
        policies = json.loads(original["policies_json"])
        policies["buffer"]["implementation"] = {
            "kind": "model",
            "model": {"source": f"/models/copy-{identity}"},
        }
        policies["buffer"]["resolved_model"] = {
            "source": f"/runs/new/inputs/{identity}/dependencies/buffer",
            "original_source": f"/models/copy-{identity}",
            "metadata": {"provider": "rllib.resource_ppo", "observation": None},
            "package_sha256": "same-weights",
        }
        entries.append(
            {
                "experiment_id": identity,
                "prepared": {**original, "policies_json": json.dumps(policies)},
            }
        )
    groups, mapping = batch._groups(entries, online=True)
    assert len(groups) == 1 and mapping["a"] == mapping["b"]
    assert batch._groups(entries)[1]["a"] != batch._groups(entries)[1]["b"]
    changed = copy.deepcopy(entries)
    policies["buffer"]["resolved_model"]["package_sha256"] = "different-weights"
    changed[1]["prepared"]["policies_json"] = json.dumps(policies)
    assert len(batch._groups(changed, online=True)[0]) == 2


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


def test_online_setup_has_no_disposable_probe_and_freezes_the_authored_layout(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from smartsom.experiments.tuning_resources import ResourceSnapshot

    original = prepared()
    entries = tuple(
        {
            "experiment_id": name,
            "prepared": asdict(original),
            "control_spec": {},
            "baseline_concurrency": 3,
        }
        for name in ("a", "b", "c")
    )
    inputs = batch.BatchInputs(
        entries,
        calibration_level="online",
        active_limit=0,
        output_root=str(tmp_path / "runs"),
    )
    root, plan, _ = batch.allocate_batch(inputs)
    monkeypatch.setattr(
        batch,
        "CalibrationMonitor",
        lambda *_args, **_kwargs: pytest.fail(
            "online must not use isolated calibration"
        ),
    )
    result = batch.calibrate(
        root,
        plan,
        monitor=SimpleNamespace(
            snapshot=lambda **kwargs: ResourceSnapshot(
                8, 16 * 1024**3, 12 * 1024**3, external_cpu_load=0
            )
        ),
    )
    assert result["calibration_level"] == "online" and not result["calibrated"]
    assert result["measurements"] == [] and result["wall_seconds"] == 0
    assert result["groups"] == batch._groups(entries, online=True)[1]
    selected = next(iter(result["recommendations"].values()))
    assert selected["threads"] == original.config.runtime.numerical_threads
    assert selected["num_envs"] == original.config.runtime.num_envs
    assert selected["sampling_processes"] == original.config.runtime.sampling_processes
    assert selected["concurrency"] == 3
    assert result["online_context"] and not (root / "calibration/probes").exists()


def test_online_history_uses_author_output_scope_across_nested_runs(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from smartsom.experiments import performance_profiles
    from smartsom.experiments.tuning_resources import ResourceSnapshot

    captured = []
    monkeypatch.setattr(
        performance_profiles,
        "select_online",
        lambda output, **kwargs: captured.append(output),
    )
    monitor = SimpleNamespace(
        snapshot=lambda **kwargs: ResourceSnapshot(
            8, 16 * 1024**3, 12 * 1024**3, external_cpu_load=0
        )
    )
    entries = (
        {"experiment_id": "a", "prepared": asdict(prepared()), "control_spec": {}},
    )
    stable = str(tmp_path / "custom-output")
    for name in ("run-one", "run-two"):
        inputs = batch.BatchInputs(
            entries,
            calibration_level="online",
            active_limit=0,
            output_root=str(tmp_path / "custom-output" / name / "performance"),
            provenance={"kind": "author-plan", "performance_root": stable},
        )
        root, plan, _ = batch.allocate_batch(inputs)
        batch.calibrate(root, plan, monitor=monitor)
        assert batch._performance_output(plan) == stable
    assert captured == [stable, stable]


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


def test_utf8_ledger_failure_records_failed_status_under_legacy_locale(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    inputs = batch.BatchInputs(
        ({"experiment_id": "a", "prepared": asdict(prepared()), "control_spec": {}},),
        output_root=str(tmp_path),
    )
    root, plan, state = batch.allocate_batch(inputs)
    state["note"] = "调度 café"
    write_json(root / "batch.json", state)
    original = Path.read_text

    def legacy_locale(path, encoding=None, errors=None):
        return original(path, encoding=encoding or "gbk", errors=errors)

    monkeypatch.setattr(Path, "read_text", legacy_locale)

    def fail(*args, **kwargs):
        raise RuntimeError("worker failed 调度")

    monkeypatch.setattr(batch, "calibrate", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        batch.execute_batch(root, plan, state)
    actual = json.loads(original(root / "batch.json", encoding="utf-8"))
    assert actual["note"] == "调度 café"
    assert actual["status"] == "failed"
    assert actual["failure"] == {
        "exception": "RuntimeError",
        "message": "worker failed 调度",
    }


@pytest.mark.parametrize(
    "outcome",
    [
        "stop",
        "late_stop",
        "no_stop",
        "unfinished",
        "error_stop",
        "raised_error_stop",
        "prior_error_stop",
        "prior_error_callback_stop",
    ],
)
def test_terminal_fit_reconciles_fresh_stop_without_hiding_worker_failure(
    tmp_path, monkeypatch, outcome
):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from smartsom.experiments import control, tuning_resources

    original = prepared()
    multiple = outcome in {"prior_error_stop", "prior_error_callback_stop"}
    entries = [{"experiment_id": "a", "prepared": asdict(original), "control_spec": {}}]
    if multiple:
        config = json.loads(original.config_json)
        config["runtime"]["device"] = "cuda"
        cuda = replace(original, config_json=json.dumps(config))
        entries.append(
            {"experiment_id": "b", "prepared": asdict(cuda), "control_spec": {}}
        )
    root, plan, state = batch.allocate_batch(
        batch.BatchInputs(
            tuple(entries),
            calibration_level="off",
            active_limit=0,
            execution="fixed",
            output_root=str(tmp_path),
        )
    )
    (root / "control").mkdir()
    write_json(root / "control/owner.json", {"id": "run", "status": "running"})
    monkeypatch.setattr(control.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(control, "_REQUEST_CACHE", {})
    assert control.requested(root) is False
    shutdown = []
    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: None,
        shutdown=lambda: shutdown.append(True),
        _private=SimpleNamespace(ray_constants=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(
        sys.modules,
        "smartsom.experiments.tuning_broker",
        SimpleNamespace(
            AdaptiveBroker=lambda *args, **kwargs: SimpleNamespace(
                allocation=lambda *args: None
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "smartsom.experiments.tuning_callbacks",
        SimpleNamespace(EvidenceCallback=lambda *args: None),
    )
    monkeypatch.setattr(
        tuning_resources,
        "ResourceMonitor",
        lambda: SimpleNamespace(
            snapshot=lambda **kwargs: SimpleNamespace(cpus=2, gpus=[])
        ),
    )

    calls = []
    worker_error = RuntimeError("real worker error")

    def fit(device):
        calls.append(device)
        saved = json.loads((root / "batch.json").read_text())
        row = saved["entries"]["b" if device == "cuda" else "a"]
        real_error = device == "cpu" if multiple else "error" in outcome
        row.update(
            status="completed" if outcome in {"no_stop", "late_stop"} else "running",
            updates=1,
            physical_ticks=4,
            checkpoint="committed",
        )
        if real_error:
            row.update(
                status="failed",
                failure={"exception": "RuntimeError", "message": "real worker error"},
            )
        write_json(root / "batch.json", saved)
        if (multiple and device == "cuda") or outcome in {
            "stop",
            "late_stop",
            "error_stop",
            "raised_error_stop",
        }:
            write_json(root / "control/stop.json", {"id": "run"})
            # Another process has a distinct cache and sees the request, while the
            # driver's cached negative remains younger than 200 ms at fit return.
            with monkeypatch.context() as worker:
                worker.setattr(control, "_REQUEST_CACHE", {})
                assert control.requested(root) is True
            assert control.requested(root) is False
        if multiple and device == "cuda" and outcome == "prior_error_callback_stop":
            raise control.StopRequested("later callback stop")
        if outcome == "raised_error_stop":
            raise worker_error
        return SimpleNamespace(errors=[worker_error] if real_error else [])

    def build_tuner(cohort, **kwargs):
        for entry in cohort:
            assert entry["evaluation_case_root"] == str(
                root / "experiments" / entry["experiment_id"] / ".evaluation-recovery"
            )
        return SimpleNamespace(fit=lambda: fit(kwargs["device"]))

    monkeypatch.setitem(
        sys.modules,
        "smartsom.experiments.tuning_ray",
        SimpleNamespace(build_tuner=build_tuner),
    )
    if multiple:
        with pytest.raises(BaseException) as failure:
            batch._execute_batch(root, plan, state)
        assert failure.value is worker_error
        assert isinstance(failure.value.__cause__, control.StopRequested)
        assert calls == ["cpu", "cuda"]
    elif outcome in {"stop", "late_stop"}:
        with pytest.raises(control.StopRequested):
            batch._execute_batch(root, plan, state)
    elif outcome == "raised_error_stop":
        with pytest.raises(RuntimeError, match="real worker error"):
            batch._execute_batch(root, plan, state)
    else:
        result = batch._execute_batch(root, plan, state)
        assert result["status"] == ("completed" if outcome == "no_stop" else "failed")
    saved = json.loads((root / "batch.json").read_text())
    assert saved["status"] == (
        "interrupted"
        if outcome in {"stop", "late_stop"}
        else "completed"
        if outcome == "no_stop"
        else "failed"
    )
    assert saved["entries"]["a"]["checkpoint"] == "committed"
    assert saved["entries"]["a"]["updates"] == 1
    if outcome == "stop":
        assert saved["entries"]["a"]["status"] == "interrupted"
    if outcome == "late_stop":
        assert saved["entries"]["a"]["status"] == "completed"
    if "error" in outcome:
        assert saved["entries"]["a"]["failure"]["message"] == "real worker error"
    if multiple:
        assert saved["failure"] == {
            "exception": "RuntimeError",
            "message": "real worker error",
        }
        assert saved["entries"]["a"]["status"] == "failed"
        assert saved["entries"]["b"]["status"] == "interrupted"
        assert saved["segments"][0]["status"] == "failed"
    assert shutdown == [True]
