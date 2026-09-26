"""Active-budget and all-group gates without running a training framework."""

from dataclasses import replace

from smartsom.experiments.tuning_calibration import (
    CalibrationController,
    CandidateMeasurement,
    generate_candidates,
    integer_candidates,
    rank_measurements,
)
from smartsom.experiments.tuning_resources import (
    GIB,
    ExecutionProfile,
    GPUInfo,
    ResourceSnapshot,
)

BASE = ExecutionProfile(1, 1, "cpu")


def snapshot(**changes):
    return ResourceSnapshot(
        **(
            {
                "cpus": 8.0,
                "memory_total": 16 * GIB,
                "memory_available": 12 * GIB,
                "external_cpu_load": 0.0,
            }
            | changes
        )
    )


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class Monitor:
    def __init__(self, observations):
        self.observations = list(observations)
        self.last = self.observations[-1]

    def snapshot(self, exclude_pids=()):
        if self.observations:
            self.last = self.observations.pop(0)
        return self.last


def test_power_candidates_include_allocation_boundary_and_joint_cpu_limit():
    assert integer_candidates(11) == (1, 2, 4, 8, 11)
    assert integer_candidates(0.9) == ()
    candidates = generate_candidates(snapshot(), mode="throughput")
    assert candidates[0] == BASE
    assert ExecutionProfile(1, 7, "cpu") in candidates
    assert all(p.threads * p.concurrency <= 7 for p in candidates)


def test_ranking_is_batch_throughput_and_filters_invalid_or_peak_over_budget():
    good = CandidateMeasurement(BASE, 10.0, GIB, {"sample": 1.0})
    faster = CandidateMeasurement(ExecutionProfile(1, 2), 18.0, 2 * GIB)
    oom = CandidateMeasurement(ExecutionProfile(1, 4), 40.0, 16 * GIB)
    failed = replace(good, throughput=100, valid=False, reason="worker error")
    nan = replace(good, throughput=float("nan"))
    assert rank_measurements(
        [failed, oom, nan, good, faster], snapshot=snapshot(), mode="throughput"
    ) == (faster, good)


def test_resource_wait_is_not_active_time_and_is_visible():
    clock, notices, budgets = Clock(), [], []
    monitor = Monitor([snapshot(external_cpu_load=100.0)] * 4 + [snapshot()])

    def probe(group, profile, remaining):
        assert group == "small"
        budgets.append(remaining)
        clock.sleep(3)
        return CandidateMeasurement(profile, 12.0, GIB)

    result = CalibrationController(
        probe, monitor, clock=clock, sleeper=clock.sleep
    ).run({"small": "small"}, {"small": [BASE]}, on_wait=notices.append)
    assert result.ready and result.active_seconds == 9 and result.waiting_seconds == 3
    assert budgets == [600, 597, 594]
    assert result.converged_groups == ("small",)
    assert len(notices) == 3


def test_indefinite_resource_wait_can_cancel_without_a_probe():
    clock = Clock()
    monitor = Monitor([snapshot(external_cpu_load=100)])
    calls = []
    result = CalibrationController(
        lambda *args: calls.append(args), monitor, clock=clock, sleeper=clock.sleep
    ).run({"small": "small"}, {"small": [BASE]}, cancelled=lambda: clock.value >= 5)
    assert result.status == "cancelled" and not result.ready
    assert result.active_seconds == 0 and result.waiting_seconds == 5 and not calls


def test_optional_expansion_blocked_by_current_load_does_not_stall_batch():
    clock, calls = Clock(), []
    large = ExecutionProfile(4, 1)

    def probe(group, profile, remaining):
        calls.append(profile)
        clock.sleep(1)
        return CandidateMeasurement(profile, 10, GIB)

    report = CalibrationController(
        probe,
        Monitor([snapshot(external_cpu_load=60)]),
        clock=clock,
        sleeper=clock.sleep,
        mode="throughput",
    ).run({"a": "a"}, {"a": [BASE, large]})
    assert report.ready and report.recommendations["a"] == BASE
    assert calls == [BASE, BASE, BASE]
    assert report.waiting_seconds == 0
    assert any(
        not m.valid and "current resource load" in m.reason for m in report.measurements
    )


def test_baseline_all_groups_before_optimization_and_failure_blocks_auto_run():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append((group, profile))
        clock.sleep(2)
        return CandidateMeasurement(
            profile,
            10.0,
            GIB,
            valid=group != "large",
            reason="unsupported backend" if group == "large" else None,
        )

    candidates = {name: [BASE, ExecutionProfile(1, 2)] for name in ("small", "large")}
    result = CalibrationController(
        probe, Monitor([snapshot()]), clock=clock, sleeper=clock.sleep
    ).run({"small": "small", "large": "large"}, candidates)
    assert calls[:2] == [("small", BASE), ("large", BASE)]
    assert ("large", ExecutionProfile(1, 2)) not in calls
    assert not result.ready and result.missing_groups == ("large",)


def test_active_deadline_does_not_automatically_skip_unmeasured_group():
    clock = Clock()

    def probe(group, profile, remaining):
        clock.sleep(remaining)
        return CandidateMeasurement(profile, 10.0, GIB)

    result = CalibrationController(
        probe, Monitor([snapshot()]), clock=clock, sleeper=clock.sleep
    ).run({"small": "small", "large": "large"}, {"small": [BASE], "large": [BASE]})
    assert result.active_seconds == 600 and result.status == "deadline"
    assert not result.ready and result.missing_groups == ("large",)


def test_overrunning_callback_cannot_claim_deadline_enforcement():
    clock = Clock()

    def probe(group, profile, remaining):
        clock.sleep(remaining + 10)
        return CandidateMeasurement(profile, 10, GIB)

    result = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"small": "small"}, {"small": [BASE]}
    )
    assert not result.ready
    assert "supervisor isolation required" in result.measurements[0].reason
    assert result.active_seconds == 610


def test_supervisor_receives_remaining_budget_and_cancellation_callback():
    clock, captured = Clock(), []

    class Supervisor:
        def run(self, probe, group, profile, remaining, cancelled):
            captured.append((group, remaining, cancelled()))
            clock.sleep(remaining)
            raise TimeoutError("isolated probe terminated and joined")

    result = CalibrationController(
        lambda *a: None,
        Monitor([snapshot()]),
        active_limit=5,
        clock=clock,
        supervisor=Supervisor(),
    ).run({"small": "small"}, {"small": [BASE]})
    assert captured == [("small", 5, False)]
    assert not result.ready and result.active_seconds == 5
    assert "TimeoutError" in result.measurements[0].reason


def test_candidate_larger_than_idle_allocation_is_skipped_without_waiting():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append(profile)
        clock.sleep(1)
        return CandidateMeasurement(profile, 10, 3 * GIB)

    observation = snapshot(memory_total=8 * GIB, memory_available=8 * GIB)
    result = CalibrationController(
        probe, Monitor([observation]), clock=clock, mode="throughput"
    ).run({"small": "small"}, {"small": [BASE, ExecutionProfile(1, 4)]})
    assert result.ready and calls == [BASE, BASE, BASE]
    assert result.waiting_seconds == 0
    assert any(
        m.reason == "candidate exceeds static resource allocation"
        for m in result.measurements
    )


def test_controller_excludes_its_owned_process_tree_from_background_load():
    clock, excluded = Clock(), []

    class RecordingMonitor:
        def snapshot(self, exclude_pids=()):
            excluded.append(exclude_pids)
            return snapshot()

    result = CalibrationController(
        lambda group, profile, remaining: CandidateMeasurement(profile, 10, GIB),
        RecordingMonitor(),
        clock=clock,
        exclude_pids=(100,),
    ).run({"small": "small"}, {"small": [BASE]})
    assert result.ready and all(pids == (100,) for pids in excluded)


def test_all_group_baselines_then_expansion_then_two_leader_repeats():
    clock, calls = Clock(), []
    parallel = ExecutionProfile(1, 2)

    def probe(group, profile, remaining):
        calls.append((group, profile))
        clock.sleep(1)
        return CandidateMeasurement(profile, 20 if profile == parallel else 10, GIB)

    result = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"a": "a", "b": "b"}, {"a": [BASE, parallel], "b": [BASE, parallel]}
    )
    assert calls == [
        ("a", BASE),
        ("b", BASE),
        ("a", parallel),
        ("b", parallel),
        ("a", parallel),
        ("b", parallel),
        ("a", parallel),
        ("b", parallel),
    ]
    assert result.ready and result.active_seconds == 8
    assert result.recommendations == {"a": parallel, "b": parallel}
    assert result.converged_groups == ("a", "b") and not result.unstable_groups


def test_leader_within_five_percent_converges_after_two_repeats():
    clock = Clock()
    parallel = ExecutionProfile(1, 2)
    scores = iter((10, 20, 20.9, 20.8))

    def probe(group, profile, remaining):
        clock.sleep(1)
        return CandidateMeasurement(profile, next(scores), GIB)

    result = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"a": "a"}, {"a": [BASE, parallel]}
    )
    assert result.ready and result.status == "completed"
    assert result.active_seconds == 4 and result.converged_groups == ("a",)


def test_unstable_leader_continues_to_deadline_then_uses_baseline():
    clock = Clock()
    parallel = ExecutionProfile(1, 2)
    scores = iter((10, 20, 30, 40, 50))

    def probe(group, profile, remaining):
        clock.sleep(1)
        return CandidateMeasurement(profile, next(scores), GIB)

    result = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=5, clock=clock
    ).run({"a": "a"}, {"a": [BASE, parallel]})
    assert result.ready and result.status == "deadline" and result.active_seconds == 5
    assert result.unstable_groups == ("a",) and result.converged_groups == ()
    assert result.recommendations == {"a": BASE}
    assert (
        "baseline fallback" in result.reason
        and "convergence not established" in result.reason
    )


def test_slowdown_is_instability_and_can_eventually_confirm_new_leader():
    clock, calls = Clock(), []
    parallel = ExecutionProfile(1, 2)
    parallel_scores = iter((100, 5, 5))

    def probe(group, profile, remaining):
        calls.append(profile)
        clock.sleep(1)
        return CandidateMeasurement(
            profile, next(parallel_scores) if profile == parallel else 10, GIB
        )

    result = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"a": "a"}, {"a": [BASE, parallel]}
    )
    assert result.ready and result.recommendations == {"a": BASE}
    assert calls == [BASE, parallel, parallel, parallel, BASE, BASE]
    assert result.converged_groups == ("a",) and not result.unstable_groups


def test_expansion_reserves_two_repeat_rounds_and_exact_deadline_is_verifiable():
    clock, calls, budgets = Clock(), [], []
    parallel = ExecutionProfile(1, 2)
    expensive = ExecutionProfile(2, 2)

    def probe(group, profile, remaining):
        calls.append(profile)
        budgets.append(remaining)
        clock.sleep(1)
        return CandidateMeasurement(profile, 20 if profile == parallel else 10, GIB)

    result = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=4, clock=clock
    ).run({"a": "a"}, {"a": [BASE, parallel, expensive]})
    assert calls == [BASE, parallel, parallel, parallel]
    assert budgets == [4, 1, 2, 1]
    assert result.ready and result.active_seconds == 4
    assert result.converged_groups == ("a",) and result.recommendations["a"] == parallel


def test_deadline_without_room_for_two_repeats_keeps_measured_baseline():
    clock = Clock()

    def probe(group, profile, remaining):
        clock.sleep(1)
        return CandidateMeasurement(profile, 10, GIB)

    result = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=2, clock=clock
    ).run({"a": "a"}, {"a": [BASE, ExecutionProfile(1, 2)]})
    assert result.ready and result.recommendations == {"a": BASE}
    assert result.converged_groups == () and result.status == "deadline"
    assert "convergence not established" in result.reason


def test_group_cpu_overhead_is_counted_for_admission():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append(profile)
        clock.sleep(1)
        return CandidateMeasurement(profile, 10, GIB)

    result = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"a": {"cpu_overhead": 3}}, {"a": [BASE, ExecutionProfile(4, 1)]}
    )
    assert result.ready and calls == [BASE, BASE, BASE]
    assert any(
        m.reason == "candidate exceeds static resource allocation"
        for m in result.measurements
    )
    assert (
        rank_measurements(
            [CandidateMeasurement(ExecutionProfile(4, 1), 100, GIB)],
            snapshot=snapshot(),
            cpu_overhead=3,
        )
        == ()
    )


def test_batch_peak_is_not_multiplied_again_during_ranking():
    clock = Clock()
    concurrent = ExecutionProfile(1, 4)

    def probe(group, profile, remaining):
        clock.sleep(1)
        return CandidateMeasurement(
            profile,
            40 if profile == concurrent else 10,
            3 * GIB if profile == concurrent else GIB,
        )

    observation = snapshot(memory_total=8 * GIB, memory_available=8 * GIB)
    result = CalibrationController(probe, Monitor([observation]), clock=clock).run(
        {"a": "a"}, {"a": [BASE, concurrent]}
    )
    assert result.ready and result.recommendations == {"a": concurrent}
    assert max(m.peak_memory for m in result.measurements) == 3 * GIB


def test_gpu_candidates_and_manual_profiles_do_not_claim_concurrent_gpu_measurement():
    clock, calls = Clock(), []
    observation = snapshot(gpus=(GPUInfo("0", "test", 8 * GIB, 8 * GIB, 0.0),))
    generated = generate_candidates(observation)
    assert any(p.device == "cuda:0" for p in generated)
    assert all(p.concurrency == 1 for p in generated if p.device != "cpu")

    def probe(group, profile, remaining):
        calls.append(profile)
        clock.sleep(1)
        return CandidateMeasurement(profile, 10, GIB)

    result = CalibrationController(probe, Monitor([observation]), clock=clock).run(
        {"a": "a"}, {"a": [BASE, ExecutionProfile(1, 2, "cuda:0")]}
    )
    assert result.ready and calls == [BASE, BASE, BASE]
    assert any("concurrency=1 only" in (m.reason or "") for m in result.measurements)


def test_invalid_repeat_never_promotes_one_fast_measurement():
    clock = Clock()
    parallel = ExecutionProfile(1, 2)
    count = 0

    def probe(group, profile, remaining):
        nonlocal count
        clock.sleep(1)
        count += 1
        return CandidateMeasurement(
            profile,
            100 if profile == parallel else 10,
            GIB,
            valid=count != 3,
            reason="worker failed" if count == 3 else None,
        )

    result = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"a": "a"}, {"a": [BASE, parallel]}
    )
    assert result.ready and result.recommendations == {"a": BASE}
    assert result.converged_groups == ("a",)


def test_mixed_device_groups_all_receive_baselines_before_expansion():
    clock, calls = Clock(), []
    gpu_base = ExecutionProfile(1, 1, "cuda:0")
    gpu_larger = ExecutionProfile(2, 1, "cuda:0")
    cpu_larger = ExecutionProfile(1, 2)
    observation = snapshot(gpus=(GPUInfo("0", "test", 8 * GIB, 8 * GIB, 0.0),))

    def probe(group, profile, remaining):
        calls.append((group, profile))
        clock.sleep(1)
        return CandidateMeasurement(
            profile, 10, GIB, peak_gpu_memory=GIB if profile.device != "cpu" else 0
        )

    result = CalibrationController(probe, Monitor([observation]), clock=clock).run(
        {"cpu": "cpu", "gpu": "gpu"},
        {"cpu": [BASE, cpu_larger], "gpu": [gpu_base, gpu_larger]},
        baseline_profiles={"cpu": BASE, "gpu": gpu_base},
    )
    assert calls[:2] == [("cpu", BASE), ("gpu", gpu_base)]
    assert result.ready and result.converged_groups == ("cpu", "gpu")


def test_measure_progress_is_parent_serializable_and_tracks_cumulative_budgets():
    import json

    clock, events = Clock(), []
    monitor = Monitor([snapshot(external_cpu_load=100)] * 3 + [snapshot()])

    def probe(group, profile, remaining):
        assert events[-1]["stage"] == "measuring"
        clock.sleep(2)
        return CandidateMeasurement(profile, 10, GIB)

    result = CalibrationController(
        probe, monitor, clock=clock, sleeper=clock.sleep
    ).run(
        {"a": "a"},
        {"a": [BASE]},
        on_measure=events.append,
    )
    assert result.ready
    assert [event["stage"] for event in events] == ["measuring", "measured"] * 3
    assert [event["active_seconds"] for event in events] == [0, 2, 2, 4, 4, 6]
    assert [event["measured"] for event in events] == [0, 1, 1, 2, 2, 3]
    assert all(
        event["waiting_seconds"] == 2 and event["candidates"] == 3 for event in events
    )
    assert events[0]["profile"] == {"threads": 1, "concurrency": 1, "device": "cpu"}
    assert json.loads(json.dumps(events)) == events
