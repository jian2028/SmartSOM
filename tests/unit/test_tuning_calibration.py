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
    candidates = generate_candidates(snapshot(), mode="performance")
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
        [failed, oom, nan, good, faster], snapshot=snapshot(), mode="performance"
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
    assert budgets == [597, 594, 591]
    assert result.wall_seconds == 12
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
        mode="performance",
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
    assert ("large", ExecutionProfile(1, 2)) in calls
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


def test_directory_fair_baseline_budget_allows_next_group_after_timeout():
    clock, calls = Clock(), []

    class Supervisor:
        def run(self, probe, group, profile, remaining, cancelled):
            calls.append((group, remaining))
            if group == "slow":
                clock.sleep(remaining)
                return CandidateMeasurement(
                    profile,
                    0.0,
                    0,
                    valid=False,
                    reason="probe deadline expired",
                    termination="deadline",
                )
            clock.sleep(1)
            return CandidateMeasurement(profile, 10.0, GIB)

    report = CalibrationController(
        lambda *_: None,
        Monitor([snapshot()]),
        clock=clock,
        sleeper=clock.sleep,
        active_limit=600,
        supervisor=Supervisor(),
        fair_baselines=True,
    ).run({"slow": "slow", "fast": "fast"}, {"slow": [BASE], "fast": [BASE]})
    assert calls[0][0] == "slow" and calls[0][1] < 300
    assert calls[1][0] == "fast"
    assert "fast" in report.baselines and "slow" not in report.baselines
    assert report.wall_seconds <= 600


def test_directory_baseline_resource_wait_has_per_group_deadline():
    clock, calls = Clock(), []

    class BusyThenFree:
        def snapshot(self, exclude_pids=()):
            return snapshot(external_cpu_load=100.0 if clock.value < 270 else 0.0)

    def probe(group, profile, remaining):
        calls.append(group)
        clock.sleep(1)
        return CandidateMeasurement(profile, 10.0, GIB)

    report = CalibrationController(
        probe,
        BusyThenFree(),
        clock=clock,
        sleeper=clock.sleep,
        active_limit=600,
        fair_baselines=True,
    ).run({"blocked": "blocked", "fast": "fast"}, {"blocked": [BASE], "fast": [BASE]})
    assert calls and calls[0] == "fast"
    assert "fast" in report.baselines and "blocked" not in report.baselines
    assert report.measurements[0].termination == "deadline"
    assert report.waiting_seconds == 270


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
        probe, Monitor([observation]), clock=clock, mode="performance"
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
    assert result.recommendations == {"a": parallel}
    assert "without convergence" in result.reason


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
    assert "without convergence" in result.reason


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


def test_load_blocks_previous_leader_then_repeats_feasible_baseline():
    clock, calls = Clock(), []
    larger = ExecutionProfile(3, 1)

    def probe(group, profile, remaining):
        calls.append(profile)
        clock.sleep(1)
        return CandidateMeasurement(profile, 20 if profile == larger else 10, GIB)

    # Initial snapshot, baseline, expansion, then a loaded leader repeat.
    loaded = snapshot(external_cpu_load=55.0)
    monitor = Monitor([snapshot(), snapshot(), snapshot(), loaded])
    report = CalibrationController(
        probe, monitor, clock=clock, sleeper=clock.sleep
    ).run({"a": "a"}, {"a": [BASE, larger]})
    assert report.ready and report.status == "completed"
    assert report.recommendations == {"a": BASE}
    assert report.converged_groups == ("a",)
    assert calls == [BASE, larger, BASE, BASE]
    assert any(
        m.profile == larger
        and m.reason == "current resource load exceeds candidate budget; skipped"
        for m in report.measurements
    )


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
    assert events[0]["profile"] == {
        "threads": 1,
        "concurrency": 1,
        "device": "cpu",
        "num_envs": 1,
        "sampling_processes": 0,
    }
    assert json.loads(json.dumps(events)) == events


def test_joined_probe_deadline_preserves_previous_full_baseline_before_limit():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append(remaining)
        if len(calls) == 1:
            clock.sleep(1)
            return CandidateMeasurement(profile, 10, GIB)
        clock.sleep(remaining - 0.25)  # joined supervisor's shutdown reserve
        return CandidateMeasurement(
            profile,
            1000,
            GIB,
            valid=False,
            reason="localized watchdog message",
            termination="deadline",
        )

    report = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=5, clock=clock
    ).run({"a": "a"}, {"a": [BASE]})
    assert calls == [5, 4]
    assert report.status == "deadline" and report.active_seconds == 4.75
    assert report.ready and report.missing_groups == ()
    assert report.recommendations == {"a": BASE}
    assert report.baselines["a"].throughput == 10
    assert not report.measurements[-1].eligible
    assert report.converged_groups == () and "without convergence" in report.reason


def test_short_remaining_budget_does_not_start_hopeless_confirmation_probe():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append(remaining)
        clock.sleep(3)
        return CandidateMeasurement(profile, 10.0, GIB)

    report = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=5, clock=clock
    ).run({"a": "a"}, {"a": [BASE]})
    assert calls == [5]
    assert report.status == "deadline" and report.ready
    assert report.recommendations == {"a": BASE}
    assert len(report.measurements) == 1


def test_probe_deadline_cannot_establish_incomplete_or_unmeasured_baseline():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append(group)
        clock.sleep(remaining - 0.25)
        return CandidateMeasurement(
            profile,
            1000,
            GIB,
            valid=False,
            termination="deadline",
        )

    report = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=5, clock=clock
    ).run({"a": "a", "b": "b"}, {"a": [BASE], "b": [BASE]})
    assert calls == ["a"] and report.status == "deadline"
    assert not report.ready and report.missing_groups == ("a", "b")
    assert report.baselines == report.recommendations == {}
    assert "no valid measured candidate" in report.reason


def test_expansion_deadline_keeps_reserved_repeats_for_measured_baseline():
    clock, calls = Clock(), []
    parallel = ExecutionProfile(1, 2)

    def probe(group, profile, remaining):
        calls.append((profile, remaining))
        if profile == parallel:
            clock.sleep(remaining - 0.25)
            return CandidateMeasurement(
                profile,
                1000,
                GIB,
                valid=False,
                termination="deadline",
            )
        clock.sleep(1)
        return CandidateMeasurement(profile, 10, GIB)

    report = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=6, clock=clock
    ).run({"a": "a"}, {"a": [BASE, parallel]})
    assert calls == [(BASE, 6), (parallel, 3), (BASE, 2.25), (BASE, 1.25)]
    assert report.ready and report.recommendations == {"a": BASE}
    assert report.converged_groups == ("a",) and report.active_seconds == 5.75


def test_deadline_repeat_never_promotes_fast_unconfirmed_candidate():
    clock, calls = Clock(), []
    parallel = ExecutionProfile(1, 2)

    def probe(group, profile, remaining):
        calls.append(profile)
        if len(calls) == 3:
            clock.sleep(remaining - 0.25)
            return CandidateMeasurement(
                profile,
                1000,
                GIB,
                valid=False,
                termination="deadline",
            )
        clock.sleep(1)
        return CandidateMeasurement(profile, 100 if profile == parallel else 10, GIB)

    report = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=5, clock=clock
    ).run({"a": "a"}, {"a": [BASE, parallel]})
    assert calls == [BASE, parallel, parallel]
    assert report.ready and report.status == "deadline"
    assert report.recommendations == {"a": parallel} and report.converged_groups == ()


def test_worker_error_with_deadline_in_message_still_blocks_baseline():
    clock, calls = Clock(), []

    def probe(group, profile, remaining):
        calls.append(profile)
        if len(calls) == 1:
            clock.sleep(1)
            return CandidateMeasurement(profile, 10, GIB)
        clock.sleep(remaining)
        return CandidateMeasurement(
            profile,
            0,
            GIB,
            valid=False,
            reason="worker bug mentions deadline",
        )

    report = CalibrationController(
        probe, Monitor([snapshot()]), active_limit=5, clock=clock
    ).run({"a": "a"}, {"a": [BASE]})
    assert report.status == "deadline" and not report.ready
    assert report.recommendations == {} and report.missing_groups == ("a",)


def test_structured_cancellation_never_automatically_launches_baseline():
    clock = Clock()

    def probe(group, profile, remaining):
        clock.sleep(1)
        return CandidateMeasurement(
            profile,
            0,
            GIB,
            valid=False,
            termination="cancelled",
        )

    report = CalibrationController(probe, Monitor([snapshot()]), clock=clock).run(
        {"a": "a"}, {"a": [BASE]}
    )
    assert report.status == "cancelled" and not report.ready
    assert report.baselines == report.recommendations == {}
