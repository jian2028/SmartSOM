"""Phase clocks, resume baselines and constrained completion estimates."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from smartsom.telemetry.estimates import PhaseMeter, batch_remaining, remaining_service
from smartsom.telemetry.execution_metrics import RollingThroughput, resource_fields


def row(ticks, stage="sampling", status="running", **values):
    return {
        "id": "one",
        "status": status,
        "stage": stage,
        "unit": "physical ticks",
        "completed": ticks,
        "total": 1000,
        "values": {"physical_ticks": ticks, **values},
    }


def observe(meter, ticks, now, stage="sampling", status="running", **values):
    meter.observe(row(ticks, stage, status, **values), now=now, wall_time=1000 - now)
    return meter.snapshot()


def test_restored_counters_do_not_become_fresh_work():
    meter = PhaseMeter()
    observe(meter, 900, 0)
    data = observe(meter, 910, 10)
    assert data["training_ticks"] == 10
    assert data["phases"]["training"]["rate"] == 1
    assert data["phases"]["training"]["seconds"] == 10
    assert "monotonic" not in str(data)


def test_validation_wall_time_is_not_charged_to_training_rate():
    meter = PhaseMeter()
    observe(meter, 0, 0)
    observe(
        meter,
        100,
        10,
        "validation",
        validation_tick=0,
        validation_round=1,
        validation_finished=0,
        data_seed=1,
    )
    observe(
        meter,
        100,
        40,
        "validation",
        validation_tick=300,
        validation_round=1,
        validation_finished=1,
        validation_case_active=False,
        data_seed=1,
    )
    observe(meter, 100, 40, "sampling")
    data = observe(meter, 200, 50)
    assert data["phases"]["training"]["rate"] == 10
    assert data["phases"]["training"]["seconds"] == 20
    assert data["phases"]["validation"]["seconds"] == 30
    assert data["execution_ticks"] == 500


def test_pause_and_attempt_change_exclude_idle_time_and_old_counters():
    meter = PhaseMeter()
    observe(meter, 900, 0)
    observe(meter, 910, 10, "paused", "paused")
    observe(meter, 910, 100, "sampling")
    data = observe(meter, 920, 110)
    assert data["phases"]["training"]["rate"] == 1
    assert data["phases"]["training"]["seconds"] == 20
    meter.observe(row(900), now=120, wall_time=1, attempt="new")
    meter.observe(row(910), now=130, wall_time=999999, attempt="new")
    assert meter.snapshot()["training_ticks"] == 10
    assert meter.snapshot()["phases"]["training"]["rate"] == 1


def test_duplicate_and_out_of_order_events_never_double_count():
    meter = PhaseMeter()
    observe(meter, 0, 0)
    data = observe(meter, 100, 10)
    observe(meter, 100, 10)
    observe(meter, 200, 9)
    assert meter.snapshot()["training_ticks"] == data["training_ticks"]
    assert meter.snapshot()["phases"]["training"]["seconds"] == 10


def test_validation_metadata_does_not_reuse_prior_case_counter(monkeypatch):
    from smartsom.telemetry import execution_metrics
    from smartsom.telemetry.runtime import RuntimeDisplay

    clock = [0.0]
    monkeypatch.setattr(execution_metrics.time, "monotonic", lambda: clock[0])
    view = RuntimeDisplay(kind="training", quiet=True)
    view.update(
        "training",
        {"stage": "sampling", "physical_ticks": 0},
        unit="physical ticks",
        total=16,
    )
    for round_number in (1, 2):
        clock[0] += 1
        view.update("training", {"validation_round": round_number})
        view.update("training", {"stage": "validation"})
        view.update(
            "training",
            {
                "stage": "validation",
                "validation_tick": 0,
                "validation_finished": 0,
                "validation_case_active": True,
            },
        )
        clock[0] += 10
        view.update(
            "training",
            {
                "stage": "validation",
                "validation_tick": 8,
                "validation_finished": 1,
                "validation_case_active": False,
            },
        )
    data = view.execution_telemetry.native.snapshot()
    assert data["reason"] is None
    assert data["phases"]["validation"]["cases_done"] == 2
    assert data["execution_ticks"] == 16


def test_metadata_cannot_create_zero_baseline_before_restored_counter(monkeypatch):
    from smartsom.telemetry import execution_metrics
    from smartsom.telemetry.runtime import RuntimeDisplay

    clock = [0.0]
    monkeypatch.setattr(execution_metrics.time, "monotonic", lambda: clock[0])
    view = RuntimeDisplay(kind="training", quiet=True)
    view.update(
        "training",
        {"stage": "validation", "validation_round": 1},
        unit="physical ticks",
        total=1000,
    )
    assert view.execution_telemetry.last_training is None
    clock[0] = 10
    view.update("training", {"stage": "validation", "physical_ticks": 900})
    data = view.execution_telemetry.native.snapshot()
    assert data["training_ticks"] == 0
    assert data["execution_ticks"] == 0
    view.seed_execution(
        physical_ticks=1000, completed_phases=["training", "validation"]
    )
    view.configure(
        {
            "mode": "train-evaluate",
            "training_total": 1000,
            "validation_rounds": 2,
            "validation_horizons": [100],
            "evaluation_horizons": [100],
        }
    )
    result = view.execution_telemetry.summarize(view, now=10, wall_time=10)
    assert result["batch"]["phase_work"]["training"] == 0
    assert result["batch"]["phase_work"]["validation"] == 0
    assert result["batch"]["phase_work"]["evaluation"] == 100
    assert result["experiment"]["training_ticks"] == 0


def test_partial_plan_overlay_preserves_required_obligations():
    from smartsom.telemetry.execution_metrics import merged_plan

    full = {
        "mode": "train-evaluate",
        "validation_horizons": [100],
        "evaluation_horizons": [200],
        "eta_group": "same-job",
    }
    plan = merged_plan(full, {"control_names": ["initial"]})
    assert plan == {**full, "control_names": ["initial"]}


def test_directory_metrics_use_current_stage_and_discard_prior_tune_values(
    tmp_path, monkeypatch
):
    from smartsom.telemetry import execution_metrics
    from smartsom.telemetry.runtime import RuntimeDisplay

    clock = [0.0]
    monkeypatch.setattr(execution_metrics.time, "monotonic", lambda: clock[0])
    snapshot = SimpleNamespace(
        cpus=2,
        cpu_load=100,
        memory_available=4096,
        unavailable=(),
        observations={
            "used_cpu_cores": 4,
            "visible_cpu_total": 8,
            "host_cpu_total": 16,
        },
        processes=(),
    )
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    view.root = tmp_path
    (tmp_path / "logs").mkdir()
    view.stage = "stage_10"
    view.configure_batch([{"id": "rule", "stage": 10}])
    view.execution_telemetry.resource_monitor = SimpleNamespace(
        snapshot=lambda: snapshot
    )
    view.tuning = {
        "batch_training_active": False,
        "online": {"old": {"latest_throughput": 999}},
        "total_ticks_per_second": 999,
        "resources": {"cpus_used": 999},
    }
    view.tasks["rule"] = {
        **row(0),
        **source("a", 1, 0),
        "id": "rule",
        "name": "rule",
        "unit": "entries",
    }
    view.publish(force=True)
    assert view.tuning["native_stage"]
    assert "online" not in view.tuning
    assert view.tuning["total_ticks_per_second"] is None
    assert view.tuning["resources"]["cpus_used"] == 4
    assert view.tuning["resources"]["cpus_total"] == 8
    clock[0] = 10
    view.tasks["rule"]["execution"] = source("a", 2, 10)["execution"]
    view.publish(force=True)
    assert view.tuning["total_ticks_per_second"] == 1.2
    view.stage = "stage_20"
    view.configure_batch([{"id": "rule", "stage": 10}, {"id": "next", "stage": 20}])
    view.tasks["next"] = {
        **row(1000),
        **source("b", 1, 1000),
        "id": "next",
        "name": "next",
        "unit": "entries",
    }
    clock[0] = 11
    view.publish(force=True)
    assert view.tuning["metrics_stage"] == "stage_20"
    assert view.tuning["total_ticks_per_second"] is None
    assert view.tuning["throughput"]["status"] == "collecting"


def test_real_validation_ledger_counts_pending_trigger_once_at_resume_boundary():
    plan = {
        "mode": "training",
        "training_total": 200,
        "round_size": 100,
        "validation_updates": [1, 2],
        "validation_horizons": [100],
        "validation_rounds": 2,
    }
    task = row(
        100,
        "validation",
        validation_completed_updates=[],
        validation_pending_update=1,
        validation_finished=0,
        validation_tick=50,
        validation_tick_limit=100,
        validation_case_active=True,
    )
    observation = {
        "phases": {
            "training": {"rate": 10},
            "validation": {"rate": 10},
            "saving": {"mean_seconds": 1},
        }
    }
    result = remaining_service(task, plan, observation)
    assert result["phase_work"]["validation"] == 150
    task["values"].update(
        validation_completed_updates=[1],
        validation_finished=1,
        validation_tick=100,
        validation_case_active=False,
    )
    result = remaining_service(task, plan, observation)
    assert result["phase_work"]["validation"] == 100


def test_finished_training_cannot_zero_eta_before_final_evaluation_and_save():
    plan = {
        "mode": "train-evaluate",
        "training_total": 1000,
        "evaluation_horizons": [100],
        "control_names": ["initial"],
    }
    result = remaining_service(row(1000), plan, {"phases": {}})
    assert result["seconds"] is None
    assert result["phase_work"]["evaluation"] == result["phase_work"]["initial"] == 100
    assert any("finalization" in reason for reason in result["missing"])
    assert (
        remaining_service(row(1000, "completed", "completed"), plan, {})["seconds"] == 0
    )


def job(name, seconds, status="queued", stage=0, group=None, **extra):
    return {
        "id": name,
        "seconds": seconds,
        "known_seconds": seconds,
        "status": status,
        "stage": stage,
        "barrier_group": group,
        **extra,
    }


def test_batch_heap_preserves_long_tail_instead_of_dividing_by_parallelism():
    jobs = [job("a", 5, "running"), job("b", 100, "running"), job("c", 5), job("d", 5)]
    assert batch_remaining(jobs, limit=2)["seconds"] == 100
    assert (
        batch_remaining(jobs, limit=1)["seconds"] == 110
    )  # existing two drain before new starts
    assert jobs[0]["seconds"] == 5


def test_stage_barriers_and_dependencies_are_not_averaged_away():
    jobs = [
        job("a", 100, "running", 0, "source"),
        job("b", 1, "running", 0, "source"),
        job("c", 100, stage=1, group="source"),
        job("d", 1, stage=1, group="source"),
    ]
    assert batch_remaining(jobs, limit=2)["seconds"] == 200
    jobs[2]["barrier_group"] = None
    jobs[2]["requires"] = ["b"]
    assert batch_remaining(jobs, limit=2)["seconds"] == 101


def test_unknown_resource_resume_time_is_conditional_not_an_end_timestamp():
    result = batch_remaining([job("a", 100, "paused", conditional=True)], limit=2)
    assert result["seconds"] is None and result["conditional_seconds"] == 100
    result = batch_remaining(
        [{**job("a", 100), "seconds": None, "known_seconds": 40}], limit=2
    )
    assert result["seconds"] is None and result["known_seconds"] == 40


def source(attempt, sequence, ticks, *, updated_at=100):
    return {
        "id": attempt,
        "status": "running",
        "unit": "physical ticks",
        "execution": {
            "schema": "smartsom.execution-observation/v1",
            "attempt": attempt,
            "sequence": sequence,
            "updated_at": updated_at,
            "training_ticks": ticks,
            "execution_ticks": ticks + sequence * 2,
            "phases": {},
        },
    }


def test_throughput_baselines_restore_append_and_deduplicates_nested_projections():
    meter = RollingThroughput()
    meter.sample([source("a", 1, 900)], 0)
    assert meter.sample([source("a", 2, 910)], 10) == 1.2
    assert meter.training_rate == 1
    added = source("b", 1, 5000)
    duplicate = deepcopy(added)
    duplicate["id"] = "parent-aggregate"
    meter.sample([source("a", 3, 920), added, duplicate], 20)
    assert meter.training_rate == 1
    meter.sample([source("a", 4, 930), source("b", 2, 5010)], 30)
    assert meter.training_rate == pytest.approx(4 / 3)


def test_rule_file_aggregate_counts_execution_work_once_with_nested_native_rows():
    from smartsom.telemetry.estimates import combined_observation

    meter = RollingThroughput()

    def rows(sequence, ticks):
        child = source("native", sequence, ticks)
        child["execution"]["training_ticks"] = 0
        parent = {
            "id": "rule-file",
            "unit": "entries",
            "status": "running",
            "execution": combined_observation(
                [child["execution"]],
                attempt="file-aggregate",
                sequence=sequence,
                updated_at=100,
            ),
        }
        parent["execution"]["aggregate"] = True
        return [child, parent]

    meter.sample(rows(1, 900), 0)
    assert meter.sample(rows(2, 910), 10) == 1.2
    assert meter.training_rate == 0


def test_genuine_zero_is_distinct_from_stale_and_missing_samples():
    meter = RollingThroughput()
    data = source("a", 1, 0)
    meter.sample([data], 0, wall_time=100)
    assert meter.payload["status"] == "collecting"
    assert meter.sample([data], 10, wall_time=110) == 0
    assert meter.training_rate == 0
    assert meter.sample([data], 20, wall_time=120) is None
    assert "stale" in meter.payload["reason"]
    assert (
        meter.sample([{"id": "other", "status": "running", "values": {}}], 21) is None
    )


def test_cpu_distinguishes_host_busy_owned_zero_and_unprimed_rows():
    rows = [
        SimpleNamespace(pid=1, excluded=True, cpu_cores=0),
        SimpleNamespace(pid=2, excluded=False, cpu_cores=4),
    ]
    snapshot = SimpleNamespace(
        cpus=32,
        cpu_load=12.5,
        processes=rows,
        observations={"used_cpu_cores": 4},
        unavailable=(),
    )
    data = resource_fields(snapshot, wall_time=100)
    assert data["cpus_used"] == 4 and data["owned_cpu_cores"] == 0
    rows[0].cpu_cores = None
    data = resource_fields(snapshot)
    assert data["cpus_used"] == 4 and data["owned_cpu_cores"] is None
    assert data["owned_cpu_status"] == "collecting"
    assert resource_fields(None)["cpu_status"] == "unavailable"


def test_cpu_birth_identity_excludes_foreign_pid_reuse_and_reports_lookup_failure(
    tmp_path, monkeypatch
):
    from smartsom.experiments import control

    record = {
        "schema": "smartsom.run-control/v1",
        "status": "running",
        "owner": {"pid": 1, "created": "original"},
        "members": [],
    }
    monkeypatch.setattr(control, "read", lambda root: record)
    snapshot = SimpleNamespace(
        cpus=4,
        cpu_load=50,
        processes=[SimpleNamespace(pid=1, cpu_cores=2, excluded=False)],
        observations={"used_cpu_cores": 2},
        unavailable=(),
    )
    assert (
        resource_fields(snapshot, roots=[tmp_path], birth=lambda _: "replacement")[
            "owned_cpu_cores"
        ]
        == 0
    )
    assert (
        resource_fields(snapshot, roots=[tmp_path], birth=lambda _: "original")[
            "owned_cpu_cores"
        ]
        == 2
    )
    assert (
        resource_fields(snapshot, roots=[tmp_path], birth=lambda _: None)[
            "owned_cpu_cores"
        ]
        is None
    )

    def failed(_):
        raise OSError("sample failed")

    result = resource_fields(snapshot, roots=[tmp_path], birth=failed)
    assert result["cpus_used"] == 2 and result["owned_cpu_cores"] is None
    assert "failed" in result["owned_cpu_reason"]


def test_throughput_restart_pickle_and_wall_jump_cannot_reuse_old_window():
    import pickle

    meter = RollingThroughput()
    meter.sample([source("a", 1, 900, updated_at=100)], 0, wall_time=100)
    assert meter.sample([source("a", 2, 910, updated_at=80)], 10, wall_time=9999) == 1.2
    restored = pickle.loads(pickle.dumps(meter))
    assert restored.sample([source("a", 3, 920)], 500) is None
    assert restored.training_rate is None


def test_receiver_rejects_negative_nonfinite_and_changed_duplicate_counters():
    meter = RollingThroughput()
    meter.sample([source("a", 1, 100)], 0)
    corrupted = source("a", 1, 999)
    assert meter.sample([corrupted], 10) is None
    assert "without a source event" in meter.payload["reason"]
    bad = source("a", 2, -1)
    assert meter.sample([bad], 20) is None
    assert "invalid execution observation" in meter.payload["reason"]
    bad["execution"]["training_ticks"] = float("nan")
    assert meter.sample([bad], 21) is None


def test_render_and_publish_frequency_do_not_advance_source_phase_clock(monkeypatch):
    import time

    from smartsom.telemetry.runtime import RuntimeDisplay

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    view = RuntimeDisplay(kind="training", quiet=True)
    view.configure({"mode": "training", "training_total": 1000, "round_size": 100})
    view.update(
        "training",
        {"stage": "sampling", "physical_ticks": 900},
        total=1000,
        unit="physical ticks",
    )
    for number in range(1, 10):
        clock[0] = number
        view.publish(force=True)
        view.render()
    clock[0] = 10
    view.update(
        "training",
        {"stage": "sampling", "physical_ticks": 910},
        total=1000,
        unit="physical ticks",
    )
    data = view.snapshot()["execution"]["experiment"]
    assert data["phases"]["training"]["seconds"] == 10
    assert data["phases"]["training"]["rate"] == 1
    assert data["training_ticks"] == 10


def test_execution_contract_monitor_and_attach_are_read_only(tmp_path, monkeypatch):
    import json

    from smartsom.telemetry import monitor as module
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    view = RuntimeDisplay(kind="train-evaluate", quiet=True)
    view.bind(tmp_path)
    view.configure(
        {"mode": "train-evaluate", "training_total": 1000, "evaluation_horizons": [100]}
    )
    view.update(
        "training",
        {"stage": "sampling", "physical_ticks": 900},
        total=1000,
        unit="physical ticks",
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert (
        module.monitor(tmp_path, once=True, options=DisplayOptions(verbose=False)) == 0
    )
    keys = iter(["]", "[", "d"])
    monkeypatch.setattr(module, "_key", lambda *_: next(keys))
    assert (
        module.monitor(
            tmp_path,
            controlling=True,
            options=DisplayOptions(verbose=False, progress="off"),
        )
        == 0
    )
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    payload = json.loads((tmp_path / "logs/progress.json").read_text())
    payload["execution"]["batch"]["seconds"] = float("nan")
    (tmp_path / "logs/progress.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="estimate duration"):
        module.read_snapshot(tmp_path)
