"""Hand-computed named-rule decisions, independent of any neural vector schema."""

from dataclasses import replace
from itertools import permutations

import pytest

from smartsom.algorithms.rule_actions import project, resolve
from smartsom.domain.base_decisions import BaseCandidate, BaseDecision


def decision(group, stage, *actions):
    return BaseDecision(
        0,
        group,
        "owner",
        stage,
        tuple(BaseCandidate(name, action) for name, action in actions),
    )


def row(audit, rule, mode=None):
    return next(
        v for v in audit["rules"].values() if v["rule"] == rule and v["mode"] == mode
    )


def test_pickup_aliases_priority_and_empty_urgency():
    source = decision(
        "dispatcher", "pickup", ("near", ("a",)), ("stock", ("b",)), ("urgent", ("c",))
    )
    values = {
        "near": dict(distance=1, ready_count=0, ready_min_due=-9, commitments=4),
        "stock": dict(distance=4, ready_count=3, ready_min_due=2, commitments=2),
        "urgent": dict(distance=3, ready_count=1, ready_min_due=-2, commitments=0),
    }
    projected, audit = project(source, values)
    assert [c.action for c in projected.candidates] == [("a",), ("b",), ("c",)]
    assert row(audit, "fewest_commitments")["status"] == "alias"
    assert row(audit, "most_urgent_ready_stock")["action"] == ("c",)
    assert [c.rule_code for c in projected.candidates] == [1, 2, 3]
    for c in projected.candidates:
        assert resolve(audit, c.identity) == c.action
    with pytest.raises(ValueError, match="retained"):
        resolve(audit, next(iter(audit["alias_map"])))


def test_retained_rules_keep_identity_and_count_only_selected_rule():
    from smartsom.algorithms.rule_actions import RuleMetrics

    source = decision("machine", "process", ("job", ("j1", "normal")))
    values = {"job": dict(due_slack=2, nominal_duration=3, queue_age=4)}
    projected, audit = project(source, values, merge_aliases=False)
    assert len(projected.candidates) == 3
    assert len({c.identity for c in projected.candidates}) == 3
    assert {resolve(audit, c.identity) for c in projected.candidates} == {
        ("j1", "normal")
    }
    assert not audit["alias_map"]
    selected = projected.candidates[1]
    metrics = RuleMetrics()
    metrics.add(projected, selected.action, selected_identity=selected.identity)
    rules = metrics.summary()["machine.process"]["rules"]
    assert rules["shortest_nominal_processing:normal"]["selected"] == 1
    assert sum(r.get("selected", 0) for r in rules.values()) == 1


def test_retained_custom_rules_still_reject_illegal_targets():
    source = decision("machine", "process", ("job", ("j1", "normal")))
    with pytest.raises(ValueError, match="illegal candidate"):
        project(
            source,
            {},
            rules=("custom",),
            custom={"custom": lambda facts, request, candidates: "missing"},
            merge_aliases=False,
        )


@pytest.mark.parametrize(
    "flags,available",
    [((True, True), True), ((False, False), False), ((True, False), False)],
)
def test_delivery_machine_only_rules_use_named_destination_kind(flags, available):
    source = decision("dispatcher", "delivery", ("a", ("pre",)), ("b", ("out",)))
    values = {
        "a": dict(distance=1, pre_count=4, pre_work=8, machine_destination=flags[0]),
        "b": dict(distance=2, pre_count=1, pre_work=2, machine_destination=flags[1]),
    }
    _, audit = project(source, values)
    assert (
        row(audit, "least_pre_nominal_work")["status"] != "unavailable"
    ) == available
    if available:
        assert row(audit, "least_pre_nominal_work")["action"] == ("out",)


@pytest.mark.parametrize("group", ["post", "input"])
def test_idle_age_constant_rule_unavailable_and_stop_independent(group):
    source = decision(
        group, "accept", ("a", ("v1",)), ("b", ("v2",)), ("STOP", ("STOP",))
    )
    values = {
        "a": dict(distance=2, idle_age=3),
        "b": dict(distance=1, idle_age=3),
        "STOP": {},
    }
    projected, audit = project(source, values)
    assert row(audit, "longest_continuous_unassigned_idle")["status"] == "unavailable"
    assert [c.action for c in projected.candidates] == [("v2",), ("STOP",)]
    values["a"]["idle_age"] = 5
    projected, audit = project(source, values)
    assert row(audit, "longest_continuous_unassigned_idle")["action"] == ("v1",)


@pytest.mark.parametrize("group", ["post", "input"])
def test_service_feasibility_filters_terminal_jobs_and_arrival_breaks_score_ties(group):
    source = decision(
        group, "service", ("a", ("j1", "v1")), ("b", ("j2", "v2")), ("c", ("j3", "v3"))
    )
    values = {
        "a": dict(
            due_slack=2,
            queue_age=1,
            remaining_work=3,
            arrival_age=1,
            feasible_next_count=2,
            next_pre_work=9,
            has_feasible=True,
            terminal=False,
        ),
        "b": dict(
            due_slack=2,
            queue_age=4,
            remaining_work=1,
            arrival_age=5,
            feasible_next_count=1,
            next_pre_work=2,
            has_feasible=True,
            terminal=False,
        ),
        "c": dict(
            due_slack=9,
            queue_age=0,
            remaining_work=0,
            arrival_age=0,
            feasible_next_count=9,
            next_pre_work=0,
            has_feasible=False,
            terminal=True,
        ),
    }
    _, audit = project(source, values)
    assert row(audit, "job_edd")["action"] == ("j2", "v2")
    assert row(audit, "most_currently_feasible_next_machines")["action"] == ("j1", "v1")
    assert row(audit, "least_next_pre_nominal_work")["action"] == ("j2", "v2")
    for v in values.values():
        v["has_feasible"] = False
    _, audit = project(source, values)
    assert row(audit, "least_next_pre_nominal_work")["status"] == "unavailable"


@pytest.mark.parametrize("stage", ["accept", "service"])
def test_pre_due_and_remaining_work_are_carried_job_facts(stage):
    source = decision("pre", stage, ("a", ("v1",)), ("b", ("v2",)))
    values = {
        "a": dict(due_slack=5, distance=1, remaining_work=8, arrival_age=7),
        "b": dict(due_slack=-2, distance=4, remaining_work=2, arrival_age=2),
    }
    _, audit = project(source, values)
    assert row(audit, "job_edd")["action"] == ("v2",)
    assert row(audit, "least_remaining_nominal_work")["status"] == "alias"


def test_machine_rules_keep_modes_and_subset_codes():
    source = decision(
        "machine",
        "process",
        ("a", ("j1", "normal")),
        ("b", ("j2", "normal")),
        ("c", ("j1", "fast")),
    )
    values = {
        "a": dict(due_slack=3, nominal_duration=1, queue_age=1),
        "b": dict(due_slack=1, nominal_duration=4, queue_age=2),
        "c": dict(due_slack=3, nominal_duration=1, queue_age=1),
    }
    projected, audit = project(source, values, rules=["shortest_nominal_processing"])
    assert {c.action for c in projected.candidates} == {
        ("j1", "normal"),
        ("j1", "fast"),
    }
    assert all(c.rule_code == 2 for c in projected.candidates)
    assert row(audit, "shortest_nominal_processing", "normal")["action"] == (
        "j1",
        "normal",
    )


def test_ties_use_semantic_identity_and_not_input_order():
    source = decision("post", "accept", ("z", ("v1",)), ("a", ("v2",)), ("m", ("v3",)))
    values = {c.identity: dict(distance=1, idle_age=0) for c in source.candidates}
    for candidates in permutations(source.candidates):
        selected, _ = project(replace(source, candidates=candidates), values)
        assert selected.candidates[0].action == ("v2",)


def test_custom_rule_receives_semantic_candidates_and_must_return_legal_identity():
    source = decision("pre", "accept", ("a", ("v1",)), ("b", ("v2",)))
    observed = []

    def choose(facts, request, candidates):
        observed.append((facts, request, candidates))
        return candidates[-1].identity

    selected, audit = project(
        source,
        {},
        rules=["custom"],
        custom={"custom": choose},
        facts={"named": "facts"},
    )
    assert selected.candidates[0].action == ("v2",)
    assert observed[0][1].inputs is None
    assert not hasattr(observed[0][2][0], "features")
    with pytest.raises(ValueError, match="illegal candidate"):
        project(source, {}, rules=["custom"], custom={"custom": lambda *a: "missing"})
    with pytest.raises(ValueError, match="no available"):
        project(source, {}, rules=["custom"], custom={"custom": lambda *a: None})


def test_invalid_subsets_reprojection_and_duplicate_actions_fail():
    source = decision("post", "accept", ("a", ("v1",)))
    values = {"a": dict(distance=1, idle_age=0)}
    for rules in ([], ["missing"], ["nearest", "nearest"]):
        with pytest.raises(ValueError, match="invalid rule subset"):
            project(source, values, rules=rules)
    selected, _ = project(source, values)
    with pytest.raises(ValueError, match="already been"):
        project(selected, values)
    with pytest.raises(ValueError, match="unique"):
        project(
            replace(
                source, candidates=(*source.candidates, BaseCandidate("b", ("v1",)))
            ),
            values,
        )
