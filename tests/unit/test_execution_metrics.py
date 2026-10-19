"""Phase clocks, resume baselines and constrained completion estimates."""

import json
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


@pytest.mark.parametrize("manual", [False, True])
def test_directory_metrics_use_current_stage_and_discard_prior_tune_values(
    tmp_path, monkeypatch, manual
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
    view.stage = "stage_0"
    view.configure_batch([{"id": "prior", "stage": 0}, {"id": "rule", "stage": 10}])
    view.execution_telemetry.resource_monitor = SimpleNamespace(
        snapshot=lambda: snapshot
    )
    view.tuning = {
        "manual": manual,
        "batch_training_active": True,
        "online": {"old": {"latest_throughput": 999}},
        "total_ticks_per_second": 999,
        "resources": {"cpus_used": 999},
    }
    view.tasks["prior"] = {
        **row(900),
        **source("prior", 1, 900),
        "id": "prior",
        "name": "prior",
    }
    view.publish(force=True)
    assert not view.tuning["native_stage"]
    assert view.tuning["resources"]["cpus_used"] == 4
    view.stage = "stage_10"
    view.tuning["batch_training_active"] = False
    view.tasks["prior"]["status"] = "completed"
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
    assert not view.tuning.get("manual")
    assert view.tuning["total_ticks_per_second"] is None
    assert view.tuning["resources"]["cpus_used"] == 4
    assert view.tuning["resources"]["cpus_total"] == 8
    assert "CPU" in view.text_summary()
    clock[0] = 10
    view.tasks["rule"]["execution"] = source("a", 2, 10)["execution"]
    view.publish(force=True)
    assert view.tuning["total_ticks_per_second"] == 1.2
    assert "1.2 ticks/s" in view.text_summary()
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


def test_live_queue_metrics_keep_both_roots_without_directory_resampling(
    tmp_path, monkeypatch
):
    from smartsom.experiments import control
    from smartsom.telemetry.runtime import RuntimeDisplay

    queue, source_root = tmp_path / "queue", tmp_path / "source"
    for root, pid in [(queue, 101), (source_root, 202)]:
        (root / "control").mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "control/owner.json").write_text(
            json.dumps(
                {
                    "schema": "smartsom.run-control/v1",
                    "status": "running",
                    "owner": {"pid": pid, "created": str(pid)},
                    # Repeated source registration must still count once.
                    "members": [{"pid": 404, "created": "404"}],
                }
            )
        )
    monkeypatch.setattr(control, "_birth", lambda pid: str(pid))
    snapshot = SimpleNamespace(
        cpus=4,
        cpu_load=75,
        memory_available=4096,
        unavailable=(),
        observations={"used_cpu_cores": 3, "visible_cpu_total": 4},
        processes=(
            SimpleNamespace(pid=101, cpu_cores=1, rss=100),
            SimpleNamespace(pid=202, cpu_cores=2, rss=200),
            SimpleNamespace(pid=404, cpu_cores=0.5, rss=400),
            SimpleNamespace(pid=303, cpu_cores=0.25, rss=300),
        ),
    )
    fields = resource_fields(snapshot, roots=[queue, source_root], wall_time=100)
    assert fields["owned_cpu_cores"] == 3.5
    assert fields["owned_memory_bytes"] == 700
    assert (
        resource_fields(snapshot, roots=[queue], wall_time=100)["owned_cpu_cores"]
        == 1.5
    )
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    view.root = queue
    view.metrics_producer = "live-queue"
    view.configure_tuning(
        {
            "stage": "training",
            "manual": True,
            "batch_training_active": True,
            "entries": [],
            "resources": fields,
            "total_ticks_per_second": 17,
            "training_ticks_per_second": 11,
            "throughput": {"status": "ok", "total": 17, "training": 11},
        }
    )
    view.publish(force=True)
    assert view.execution_telemetry.resource_monitor is None
    assert view.execution_telemetry.throughput.last is None
    assert view.tuning["resources"]["owned_cpu_cores"] == 3.5
    assert view.tuning["total_ticks_per_second"] == 17
    assert "17 ticks/s" in view.text_summary()


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


def test_batch_memory_uses_owned_rss_independently_of_unprimed_cpu():
    rows = [
        SimpleNamespace(pid=1, excluded=True, cpu_cores=None, rss=2048),
        SimpleNamespace(pid=2, excluded=False, cpu_cores=4, rss=8192),
    ]
    snapshot = SimpleNamespace(
        cpus=8, cpu_load=50, processes=rows, observations={}, unavailable=()
    )
    fields = resource_fields(snapshot)
    assert fields["owned_memory_bytes"] == 2048
    assert fields["owned_cpu_cores"] is None
    rows[0].rss = None
    assert resource_fields(snapshot)["owned_memory_bytes"] is None
    assert resource_fields(None)["owned_memory_bytes"] is None


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


def test_eta_smoothing_adapts_to_speed_changes_without_changing_measured_rate():
    meter = PhaseMeter()
    observe(meter, 0, 0)
    for second in (10, 20, 30):
        observe(meter, second * 10, second)
    data = observe(meter, 350, 40)["phases"]["training"]
    assert data["rate"] == 8.75
    assert 5 < data["eta_rate"] < data["rate"]
    observe(meter, 350, 41, "paused", "paused")
    assert meter.snapshot()["phases"]["training"]["eta_rate"] is None
    observe(meter, 350, 100, "sampling")
    assert observe(meter, 400, 110)["phases"]["training"]["eta_rate"] == pytest.approx(
        5
    )


def test_total_work_progress_includes_validation_and_evaluation_and_waits_for_completion():
    from smartsom.telemetry.estimates import task_work_progress

    plan = {
        "mode": "train-evaluate",
        "training_total": 1000,
        "validation_rounds": 2,
        "validation_horizons": [100],
        "evaluation_horizons": [300],
    }
    task = row(1000, validation_batches_finished=2)
    progress = task_work_progress(task, plan)
    assert progress["total"] == 1500
    assert progress["completed"] == 1200
    assert progress["fraction"] == 0.8
    progress = task_work_progress(
        task, plan, {"completed_phases": ["training", "validation", "evaluation"]}
    )
    assert progress["fraction"] == 0.99
    task["status"] = "completed"
    assert task_work_progress(task, plan)["fraction"] == 1
    assert (
        task_work_progress(
            row(500), {"mode": "train-evaluate", "training_total": 1000}
        )["fraction"]
        is None
    )


def test_queue_completion_eta_includes_waiting_and_respects_priority_order():
    jobs = [job("running", 50, "running"), job("a", 20), job("b", 30)]
    before = batch_remaining(jobs, limit=1)
    assert before["completion_seconds"] == {"running": 50, "a": 70, "b": 100}
    after = batch_remaining([jobs[0], jobs[2], jobs[1]], limit=1)
    assert after["completion_seconds"] == {"running": 50, "b": 80, "a": 100}
    jobs[1]["conditional"] = True
    assert batch_remaining(jobs, limit=1)["completion_seconds"] == {}
    assert batch_remaining(jobs, limit=1)["conditional_completion_seconds"]["b"] == 100


def test_queue_eta_uses_runtime_capacity_priority_and_completed_peer_rates():
    from smartsom.telemetry.runtime import RuntimeDisplay

    view = RuntimeDisplay(kind="tune", quiet=True)
    view.workflow = {"mode": "training", "training_total": 1000, "eta_group": "same"}
    view.tasks = {
        key: {**row(0, status=status), "id": key}
        for key, status in [("active", "running"), ("a", "queued"), ("b", "queued")]
    }
    view.execution_telemetry.references["same"] = {
        "training": {"rate": 10, "eta_rate": None},
        "loading": {"mean_seconds": 1},
        "saving": {"mean_seconds": 1},
        "finalization": {"mean_seconds": 1},
    }
    view.tuning = {
        "runtime_control": {"mode": "manual", "limit": 1, "priority_tasks": ["b"]}
    }
    result = view.execution_telemetry.summarize(view, now=0, wall_time=0)
    times = result["batch"]["completion_seconds"]
    assert 0 < times["active"] < times["b"] < times["a"]
    assert result["tasks"]["a"]["completion_seconds"] == times["a"]
    view.tuning["runtime_control"].update(mode="adaptive", native_effective_limit=2)
    faster = view.execution_telemetry.summarize(view, now=0, wall_time=0)
    assert faster["batch"]["seconds"] < result["batch"]["seconds"]
    view.tuning["runtime_control"]["native_effective_limit"] = 0
    held = view.execution_telemetry.summarize(view, now=0, wall_time=0)
    assert held["batch"]["seconds"] is None
    assert held["batch"]["conditional_completion_seconds"]


@pytest.mark.parametrize("status", ["removed", "partial_removed", "partial_stopped"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_terminal_runtime_tasks_do_not_delay_queued_completion(status, aggregate):
    from smartsom.telemetry.runtime import RuntimeDisplay

    view = RuntimeDisplay(kind="tune", quiet=True)
    view.workflow = {"mode": "training", "training_total": 1000, "eta_group": "same"}
    retired = {**row(0, status=status), "id": "retired"}
    if aggregate:
        retired["execution"] = {
            **PhaseMeter().snapshot(),
            "aggregate": True,
            "estimate": {
                "seconds": None,
                "known_seconds": 500,
                "conditional": True,
                "missing": ["old child observation"],
            },
        }
    view.tasks = {"retired": retired, "next": {**row(0, status="queued"), "id": "next"}}
    view.execution_telemetry.references["same"] = {
        "training": {"rate": 10},
        "loading": {"mean_seconds": 1},
        "saving": {"mean_seconds": 1},
        "finalization": {"mean_seconds": 1},
    }
    view.tuning = {"runtime_control": {"mode": "manual", "limit": 1}}
    result = view.execution_telemetry.summarize(view, now=0, wall_time=0)
    del view.tasks["retired"]
    without = view.execution_telemetry.summarize(view, now=0, wall_time=0)
    assert result["batch"]["seconds"] == without["batch"]["seconds"] == 103
    assert result["tasks"]["next"]["completion_seconds"] == 103
    assert result["tasks"]["retired"]["seconds"] == 0
    assert result["tasks"]["retired"]["missing"] == []


@pytest.mark.parametrize("status", ["pausing", "removing"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_pending_safe_transition_cannot_publish_a_definite_completion_eta(
    status, aggregate
):
    task = row(0, status=status)
    plan = {"mode": "training", "training_total": 1000}
    observation = {
        "phases": {
            "training": {"rate": 10},
            "saving": {"mean_seconds": 1},
            "finalization": {"mean_seconds": 1},
        }
    }
    if aggregate:
        observation.update(
            aggregate=True, estimate={"seconds": 102, "known_seconds": 102}
        )
    service = remaining_service(task, plan, observation)
    # Even when a queued row comes first, the live worker retains its slot.
    result = batch_remaining([job("next", 3), {**task, **service}], limit=1)
    assert result["seconds"] is None
    assert result["completion_seconds"] == {}
    assert result["conditional"]
    assert result["conditional_completion_seconds"] == {"one": 102, "next": 105}


@pytest.mark.parametrize("invalid", [-1, float("nan"), {"task": -1}, {"task": "100"}])
def test_completion_eta_rejects_invalid_snapshot_values(invalid):
    from smartsom.telemetry.execution_metrics import validate_estimate

    with pytest.raises(ValueError, match="estimate duration"):
        validate_estimate({"completion_seconds": invalid})


def test_smoothed_eta_rate_is_independent_of_event_frequency():
    sparse, frequent = PhaseMeter(), PhaseMeter()
    observe(sparse, 0, 0)
    observe(frequent, 0, 0)
    for second in range(1, 61):
        ticks = second * 10 if second <= 30 else 300 + (second - 30) * 5
        observe(frequent, ticks, second)
        if second % 10 == 0:
            observe(sparse, ticks, second)
    assert sparse.snapshot()["phases"]["training"]["eta_rate"] == pytest.approx(
        frequent.snapshot()["phases"]["training"]["eta_rate"]
    )


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, "采集中"),
        (0, "0 min"),
        (59, "<1 min"),
        (60, "1 min"),
        (1200, "20 min"),
        (3540, "59 min"),
        (3599, "1 h"),
        (3600, "1 h"),
        (5400, "1.5 h"),
        (5401, "1.6 h"),
    ],
)
def test_eta_duration_uses_short_units_without_approximation_prefix(seconds, expected):
    from smartsom.telemetry.estimates import coarse_time

    assert coarse_time(seconds) == expected
