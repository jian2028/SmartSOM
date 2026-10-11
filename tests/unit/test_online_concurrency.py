"""Synthetic commit streams verify decisions without launching experiments."""

from smartsom.experiments.online_concurrency import OnlineConcurrency


def feed(controller, members, *, rate, clock, counters, windows=2):
    # One result per member establishes the new membership's baseline.
    for member in members:
        counters.setdefault(member, [0, 0])
        counters[member][1] += 1
        controller.observe(
            members,
            member,
            ticks=counters[member][0],
            updates=counters[member][1],
            now=clock,
        )
    decision = None
    for _ in range(windows):
        clock += 30
        for member in members:
            counters[member][0] += int(rate * 30 / len(members))
            counters[member][1] += 2
            outcome = controller.observe(
                members,
                member,
                ticks=counters[member][0],
                updates=counters[member][1],
                now=clock,
            )
            decision = outcome or decision
    return clock, decision


def test_real_gain_grows_one_at_a_time_and_plateau_rolls_back():
    controller = OnlineConcurrency(4)
    counters = {}
    clock, decision = feed(controller, ("a",), rate=10, clock=0, counters=counters)
    assert decision["accepted"] and controller.limit == 2
    assert decision["members"] == ["a"]
    assert all(
        window["wall_seconds"] == 30 and window["physical_ticks"] == 300
        for window in decision["windows"]
    )
    assert decision["throughput"] == 10
    assert controller.summary()["latest_throughput"] == 10
    assert controller.summary()["measured_concurrency"] == 1
    assert controller.summary()["best_throughput"] == 10
    clock, decision = feed(
        controller, ("a", "b"), rate=18, clock=clock, counters=counters
    )
    assert decision["accepted"] and controller.limit == 3
    clock, decision = feed(
        controller, ("a", "b", "c"), rate=18.3, clock=clock, counters=counters
    )
    assert not decision["accepted"] and controller.limit == 2
    _, decision = feed(controller, ("a", "b"), rate=18, clock=clock, counters=counters)
    assert decision["accepted"] and controller.limit == 2
    assert controller.summary()["best_concurrency"] == 2


def test_startup_duplicates_missing_workers_and_short_updates_cannot_grow():
    controller = OnlineConcurrency(3)
    for now in (0, 60, 120):
        assert controller.observe(("a",), "a", ticks=1000, updates=1, now=now) is None
    assert controller.limit == 1
    assert controller.observe(("a",), "a", ticks=2000, updates=2, now=121) is None
    assert controller.observe(("a",), "a", ticks=2001, updates=3, now=122) is None
    assert controller.limit == 1
    controller.limit = 2
    assert controller.observe(("a", "b"), "a", ticks=3000, updates=4, now=180) is None
    assert controller.observe(("a", "b"), "a", ticks=4000, updates=6, now=240) is None
    assert controller.history == []


def test_noisy_validation_windows_hold_and_membership_change_discards_window():
    controller = OnlineConcurrency(3)
    counters = {}
    clock, _ = feed(controller, ("a",), rate=10, clock=0, counters=counters, windows=1)
    counters["a"][0] += 600
    counters["a"][1] += 2
    assert (
        controller.observe(
            ("a",),
            "a",
            ticks=counters["a"][0],
            updates=counters["a"][1],
            now=clock + 30,
        )
        is None
    )
    assert controller.limit == 1 and controller.history == []
    controller.reset_window()
    clock, decision = feed(controller, ("a",), rate=10, clock=1000, counters=counters)
    assert decision and controller.limit == 2
    # A tail wave with fewer members than the requested limit is not evidence
    # that two workers have low throughput.
    _, decision = feed(controller, ("a",), rate=1, clock=clock, counters=counters)
    assert decision is None and controller.limit == 2


def test_cached_hint_is_revalidated_and_does_not_start_at_cached_concurrency():
    controller = OnlineConcurrency(4, hint=4)
    assert controller.limit == 1
    counters = {}
    _, decision = feed(controller, ("a",), rate=10, clock=0, counters=counters)
    assert controller.limit == 2
    assert decision["window_min_seconds"] == 15
    assert controller.summary()["cached_hint"] == 4


def test_fresh_resume_keeps_evidence_but_restarts_live_observation():
    controller = OnlineConcurrency(3)
    _, _ = feed(controller, ("a",), rate=10, clock=0, counters={})
    fresh = OnlineConcurrency(3, history=controller.history, hint=2)
    assert fresh.limit == 1 and fresh.summary()["best_concurrency"] is None
    assert fresh.summary()["latest_throughput"] is None
    assert fresh.summary()["best_throughput"] is None
    assert fresh.history == controller.history
    assert fresh.observe(("a",), "a", ticks=5000, updates=20, now=0) is None


def test_hard_ceiling_is_never_exceeded():
    controller = OnlineConcurrency(1)
    _, decision = feed(controller, ("a",), rate=10, clock=0, counters={})
    assert decision["accepted"] and controller.limit == 1
