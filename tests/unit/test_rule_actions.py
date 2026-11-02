"""Rule projection engineering contract; no learner or training execution.

Primary intended CI owner: fast (pure detached decision projection). Ownership
manifest integration is performed by the parent task, outside this file owner.
"""

import itertools
import random
from dataclasses import replace

import pytest

from smartsom.algorithms.rule_actions import (
    PROJECTION_CONTRACT,
    RULE_BITS_START,
    project,
    resolve,
)
from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
from smartsom.domain.social import SocialCandidate, SocialLink


def candidate(name, action=None, **fields):
    values = [0.0] * 48
    for field, value in fields.items():
        values[int(field)] = float(value)
    return BaseCandidate(name, (name,) if action is None else action, tuple(values))


def request(group, stage, candidates, *, tags=(), social=False):
    links = (
        tuple(
            SocialCandidate(
                c.identity,
                (
                    SocialLink(
                        "link:" + c.identity,
                        "vehicle",
                        vector(i),
                        (vector(10 + i),),
                        count=1,
                    ),
                ),
            )
            for i, c in enumerate(candidates)
        )
        if social
        else None
    )
    return BaseDecision(
        123,
        group,
        "owner",
        stage,
        tuple(candidates),
        vector(),
        ((), (), (), tuple(vector(t) for t in tags)),
        social=links,
    )


def by_rule(audit, rule, mode=None):
    return next(
        r for r in audit["rules"].values() if r["rule"] == rule and r["mode"] == mode
    )


def test_pickup_field_mapping_aliases_and_urgent_empty_stock():
    cs = [
        candidate("near", **{"0": 1, "2": 0, "3": -9, "14": 4}),
        candidate("stock", **{"0": 4, "2": 3, "3": 2, "14": 2}),
        candidate("urgent", **{"0": 3, "2": 1, "3": -2, "14": 0}),
    ]
    projected, audit = project(request("dispatcher", "pickup", cs))
    assert by_rule(audit, "nearest")["action"] == ("near",)
    assert by_rule(audit, "most_ready_stock")["action"] == ("stock",)
    assert by_rule(audit, "most_urgent_ready_stock")["action"] == ("urgent",)
    assert by_rule(audit, "fewest_commitments")["status"] == "alias"
    assert len(projected.candidates) == 3
    assert len({c.action for c in projected.candidates}) == 3
    for c in projected.candidates:
        assert resolve(audit, c.identity) == c.action
    alias = next(iter(audit["alias_map"]))
    with pytest.raises(ValueError, match="not a retained"):
        resolve(audit, alias)
    _, empty = project(request("dispatcher", "pickup", [cs[0]]))
    assert by_rule(empty, "most_urgent_ready_stock")["status"] == "unavailable"


def test_delivery_machine_queues_and_nominal_work_do_not_use_output_names():
    cs = [
        candidate("x", **{"0": 1, "1": 1, "15": 4, "42": 5}),
        candidate("y", **{"0": 2, "1": 2, "15": 0, "42": 3}),
        candidate("z", **{"0": 3, "1": 3, "15": 2, "42": 0}),
        candidate("output_name_but_machine", **{"0": 4, "1": 9, "15": 3, "42": 8}),
    ]
    _, audit = project(request("dispatcher", "delivery", cs, tags=(3, 2, 1) * 4))
    assert by_rule(audit, "fewest_pre_jobs")["action"] == ("y",)
    assert by_rule(audit, "least_pre_nominal_work")["action"] == ("z",)
    assert {r["rule"] for r in audit["rules"].values()} == {
        "nearest",
        "least_pre_nominal_work",
        "fewest_pre_jobs",
    }


def test_empty_machine_all_zero_local_fields_is_not_output():
    cs = [candidate("arbitrary-owner", **{"0": 1, "1": 1})]
    _, machine = project(request("dispatcher", "delivery", cs, tags=(1, 2, 3)))
    _, output = project(request("dispatcher", "delivery", cs, tags=(0,)))
    assert by_rule(machine, "fewest_pre_jobs")["status"] == "alias"
    assert by_rule(output, "fewest_pre_jobs")["status"] == "unavailable"
    assert by_rule(output, "nearest")["status"] == "available"


def test_mixed_or_missing_destination_tags_report_exact_limitation():
    cs = [candidate("a"), candidate("b")]
    for tags in [(), (0, 1, 2, 3)]:
        _, audit = project(request("dispatcher", "delivery", cs, tags=tags))
        row = by_rule(audit, "fewest_pre_jobs")
        assert row["status"] == "unavailable"
        assert "owner-to-station mapping" in row["reason"]


@pytest.mark.parametrize("group", ["post", "input"])
def test_accept_uses_distinct_idle_clock_and_rejects_constant_pseudo_rule(group):
    cs = [
        candidate("near", **{"0": 1, "3": 999, "23": 2}),
        candidate("idle", **{"0": 2, "3": 0, "23": 8}),
    ]
    projected, audit = project(request(group, "accept", cs))
    assert [c.action for c in projected.candidates] == [("near",), ("idle",)]
    assert by_rule(audit, "longest_continuous_unassigned_idle")["action"] == ("idle",)
    assert all("request" not in r["rule"] for r in audit["rules"].values())
    equal = [candidate("near", **{"0": 1}), candidate("far", **{"0": 2, "3": 999})]
    projected, audit = project(request(group, "accept", equal))
    assert len(projected.candidates) == 1
    assert (
        by_rule(audit, "longest_continuous_unassigned_idle")["status"] == "unavailable"
    )
    _, alias = project(
        request(
            group,
            "accept",
            [
                candidate("near", **{"0": 1, "23": 9}),
                candidate("far", **{"0": 2, "23": 1}),
            ],
        )
    )
    assert by_rule(alias, "longest_continuous_unassigned_idle")["status"] == "alias"


@pytest.mark.parametrize("stage", ["accept", "service"])
def test_pre_rules_read_carried_job_offset(stage):
    cs = [
        candidate("a", **{"0": 0, "4": 9, "5": 9, "3": 1}),
        candidate("b", **{"0": 9, "4": 0, "5": 9, "3": 2}),
        candidate("c", **{"0": 9, "4": 9, "5": 0, "3": 9}),
    ]
    _, audit = project(request("pre", stage, cs))
    assert by_rule(audit, "job_edd")["action"] == ("b",)
    assert by_rule(audit, "least_remaining_nominal_work")["action"] == ("c",)
    if stage == "service":
        assert by_rule(audit, "oldest_recorded_arrival")["action"] == ("c",)


@pytest.mark.parametrize("group", ["post", "input"])
def test_service_six_job_rules_over_legal_pairs_and_arrival_ties(group):
    cs = [
        candidate(
            "edd",
            ("j1", "v1"),
            **{"0": -5, "1": 8, "2": 1, "19": 1, "22": 1, "23": 8, "24": 1},
        ),
        candidate(
            "queue",
            ("j2", "v2"),
            **{"0": 3, "1": 7, "2": 9, "19": 2, "22": 1, "23": 8, "24": 1},
        ),
        candidate(
            "work",
            ("j3", "v3"),
            **{"0": 4, "1": 0, "2": 2, "19": 3, "22": 1, "23": 8, "24": 1},
        ),
        candidate(
            "maxwork",
            ("j4", "v4"),
            **{"0": 5, "1": 99, "2": 3, "19": 9, "22": 1, "23": 8, "24": 1},
        ),
        candidate(
            "machines",
            ("j5", "v5"),
            **{"0": 6, "1": 5, "2": 4, "19": 2, "22": 4, "23": 8, "24": 1},
        ),
        candidate(
            "prework",
            ("j6", "v6"),
            **{"0": 7, "1": 5, "2": 4, "19": 2, "22": 1, "23": 0, "24": 1},
        ),
    ]
    projected, audit = project(request(group, "service", cs))
    for rule, physical in [
        ("job_edd", "edd"),
        ("longest_job_queue", "queue"),
        ("least_remaining_nominal_work", "work"),
        ("most_remaining_nominal_work", "maxwork"),
        ("most_currently_feasible_next_machines", "machines"),
        ("least_next_pre_nominal_work", "prework"),
    ]:
        assert by_rule(audit, rule)["physical_identity"] == physical
    assert len(projected.candidates) == 6
    assert len({c.features[45:] for c in projected.candidates}) == 6
    # Same job, different legal AGVs: current port arrival beats semantic ID.
    pairs = [
        candidate("a-young", ("job", "young"), **{"19": 1}),
        candidate("z-old", ("job", "old"), **{"19": 9}),
    ]
    for perm in itertools.permutations(pairs):
        _, tied = project(request(group, "service", perm))
        assert by_rule(tied, "job_edd")["action"] == ("job", "old")
        assert by_rule(tied, "longest_job_queue")["action"] == ("job", "old")


@pytest.mark.parametrize("group", ["post", "input"])
def test_machine_specific_job_rules_exclude_terminal_and_no_feasible_next(group):
    cs = [
        candidate("terminal", ("done", "v0"), **{"22": 100, "23": 0, "24": 1, "25": 1}),
        candidate("blocked", ("blocked", "v1"), **{"22": 0, "23": 0, "24": 0}),
        candidate("valid", ("valid", "v2"), **{"22": 1, "23": 7, "24": 1}),
    ]
    _, audit = project(request(group, "service", cs))
    for rule in (
        "most_currently_feasible_next_machines",
        "least_next_pre_nominal_work",
    ):
        assert by_rule(audit, rule)["action"] == ("valid", "v2")
    _, empty = project(request(group, "service", cs[:2]))
    for rule in (
        "most_currently_feasible_next_machines",
        "least_next_pre_nominal_work",
    ):
        assert by_rule(empty, rule)["status"] == "unavailable"


def test_machine_rule_cross_legal_modes_and_mode_nominal_duration():
    cs = []
    for mode in ["fast", "normal", "slow"]:
        cs.extend(
            [
                candidate(
                    f"a:{mode}", ("job-a", mode), **{"0": 1, "2": 1, "12": 9, "14": 0.1}
                ),
                candidate(
                    f"b:{mode}", ("job-b", mode), **{"0": 2, "2": 2, "12": 1, "14": 99}
                ),
                candidate(
                    f"c:{mode}",
                    ("job-c", mode),
                    **{"0": 3, "2": 9, "12": 8, "14": 0.01},
                ),
            ]
        )
    projected, audit = project(request("machine", "process", cs))
    assert len(projected.candidates) == 9
    for mode in ["fast", "normal", "slow"]:
        assert by_rule(audit, "shortest_nominal_processing", mode)["action"] == (
            "job-b",
            mode,
        )
        assert by_rule(audit, "longest_job_queue", mode)["action"] == ("job-c", mode)
    assert all(
        "machine:process" in c.identity and "mode=value:" in c.identity
        for c in projected.candidates
    )


def test_permutations_ties_aliases_social_identity_and_rng_are_stable():
    cs = [
        candidate("semantic-b", **{"0": 1, "2": 1, "3": 1, "14": 1}),
        candidate("semantic-a", **{"0": 1, "2": 1, "3": 1, "14": 1}),
    ]
    original = request("dispatcher", "pickup", cs, social=True)
    rng = random.getstate()
    baseline, baseline_audit = project(original)
    for perm in itertools.permutations(original.candidates):
        result, audit = project(
            replace(original, candidates=perm, social=original.social[::-1])
        )
        assert result == baseline and audit == baseline_audit
    assert random.getstate() == rng
    physical = {s.identity: s for s in original.social}
    for projected, metadata in zip(
        baseline.social, baseline_audit["selected_candidates"], strict=True
    ):
        assert projected.identity == metadata["identity"]
        assert projected.links == physical[metadata["physical_identity"]].links
    assert [c.action for c in baseline.candidates] == [("semantic-a",)]


@pytest.mark.parametrize("action", [("STOP",), ("EMPTY_WAIT",)])
def test_special_actions_remain_distinct(action):
    projected, audit = project(
        request("pre", "accept", [candidate("special", action)], social=True)
    )
    assert projected.candidates[0].action == action
    assert resolve(audit, "special") == action
    assert not audit["alias_map"]
    assert projected.social[0].identity == "special"


def test_empty_requests_and_padding_contract():
    with pytest.raises(ValueError, match="no available rules"):
        project(request("pre", "accept", []))
    source = request("dispatcher", "pickup", [candidate("a", **{"44": 1.25})])
    projected, audit = project(source)
    assert audit["contract"] == PROJECTION_CONTRACT
    assert all(len(c.features) == 48 for c in projected.candidates)
    assert (
        projected.candidates[0].features[:RULE_BITS_START]
        == source.candidates[0].features[:RULE_BITS_START]
    )
    assert projected.candidates[0].features[45:] == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="already been"):
        project(projected)
    with pytest.raises(ValueError, match="padding"):
        project(request("dispatcher", "pickup", [candidate("a", **{"45": 1})]))


def test_wrong_social_keys_duplicate_actions_and_invalid_fields_reject():
    source = request("pre", "accept", [candidate("a")], social=True)
    with pytest.raises(ValueError, match="social candidate"):
        project(replace(source, social=(replace(source.social[0], identity="wrong"),)))
    with pytest.raises(ValueError, match="must be unique"):
        project(
            replace(
                source,
                candidates=(
                    source.candidates[0],
                    replace(source.candidates[0], identity="b"),
                ),
            )
        )
    with pytest.raises(ValueError, match="nonfinite"):
        project(request("pre", "accept", [candidate("a", **{"0": float("nan")})]))


def test_original_protocol_feature_layout_and_mode_rounding():
    from smartsom.config.base_layout import base_factory
    from smartsom.domain.production import Demand, ProductionScenario, ProductionStep
    from smartsom.engine.production import ProductionSimulator

    demands = (Demand("job", (ProductionStep("op", "A", 7),), due_at=100),)
    sim = ProductionSimulator(
        ProductionScenario(
            base_factory("small"), demands, tick_limit=200, processing_rounding="ceil"
        ),
        contract="base",
    )
    p = sim.protocol
    job = next(iter(sim.jobs))
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    sim._remove(job)
    sim._place(job, sim.pre[machine], "pool")
    sim.tick = 20
    sim.jobs[job]["since"] = 10
    j = p.job_vector(job)
    assert j[0] == 0.4 and j[1] == 7 / 200 and j[2] == 0.05
    generator = p.decisions()
    physical = next(generator)
    assert (physical.group, physical.stage) == ("machine", "process")
    for c in physical.candidates:
        assert c.features[12] == p.nominal(job, machine, c.action[1]) / 200
        assert c.features[16:] == (0.0,) * 32
    projected, _ = project(physical)
    assert {c.action for c in projected.candidates} == {
        c.action for c in physical.candidates
    }
    generator.close()
    assert {v[0] for v in p.station_vectors(sim.pre[machine])} == {1.0, 2.0, 3.0}
    assert {v[0] for v in p.station_vectors("output")} == {0.0}


def test_explicit_mode_overrides_subset_codes_and_no_fallback():
    from smartsom.config.base_diagnostic import BaseAlgorithm
    from tests.unit.test_algorithm_components import recipe

    data = recipe()
    del data["action_representation"]
    with pytest.raises(ValueError, match="action_representation"):
        BaseAlgorithm.model_validate(data)
    data["action_representation"] = {
        "default": "direct",
        "roles": {"dispatcher": "rules"},
        "stages": {"dispatcher.delivery": "direct"},
        "rules": {"dispatcher.pickup": ["most_ready_stock", "nearest"]},
    }
    options = BaseAlgorithm.model_validate(data)
    assert options.action_representation.mode("dispatcher.pickup") == "rules"
    assert options.action_representation.mode("dispatcher.delivery") == "direct"
    assert options.action_representation.mode("machine.process") == "direct"
    source = request("dispatcher", "pickup", [candidate("one", **{"2": 1})])
    projected, audit = project(source, rules=("most_ready_stock", "nearest"))
    assert projected.candidates[0].features[45:] == (0, 1, 0)
    assert by_rule(audit, "nearest")["status"] == "alias"
    with pytest.raises(ValueError, match="no available"):
        project(
            request("post", "accept", [candidate("a")]),
            rules=("longest_continuous_unassigned_idle",),
        )


def test_custom_rule_freeze_and_illegal_result(tmp_path):
    import shutil

    from smartsom.algorithms.base_policy import build_policy
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import compile_reservation
    from smartsom.domain.public_facts import freeze
    from tests.unit.test_algorithm_components import author, recipe

    bundle = tmp_path / "custom"
    bundle.mkdir()
    (bundle / "helper.py").write_text(
        "def select(candidates): return candidates[-1].identity\n"
    )
    (bundle / "entry.py").write_text("""from .helper import select

def create_rule():
    def choose(facts, decision, candidates):
        assert facts['tick'] == decision.tick
        return select(candidates)
    return choose
create_rule.CONTRACT = "smartsom.rule/v1"
""")
    data = recipe()
    data["action_representation"] = {
        "default": "direct",
        "stages": {"dispatcher.pickup": "rules"},
        "rules": {"dispatcher.pickup": ["custom_last"]},
        "custom_rules": {
            "custom_last": {
                "path": "custom/entry.py",
                "source_root": "custom",
                "factory": "create_rule",
            }
        },
    }
    path = author(tmp_path, data)
    plan = compile_reservation(path)
    assert "action_representation.custom_rules.custom_last" in plan.component_report
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    entry = plan.materialize(frozen)
    shutil.rmtree(bundle)
    options = load_inputs(entry)[3]
    policy = build_policy(options, training=False)
    original = request("dispatcher", "pickup", [candidate("a"), candidate("b")])
    projected = policy.observation_transform(freeze({"tick": original.tick}), original)
    assert [c.action for c in projected.candidates] == [("b",)]
    assert policy.choose(projected) == ("b",)
    with pytest.raises(ValueError, match="illegal candidate"):
        project(original, rules=("bad",), custom={"bad": lambda *args: "ghost"})
    # Runtime never reads a modified source under its frozen digest.
    spec = options.action_representation.custom_rules["custom_last"]
    from pathlib import Path

    Path(spec.path).write_text("# changed source\n")
    with pytest.raises(ValueError, match="source mismatch"):
        build_policy(options, training=False)


def test_rule_metrics_specials_denominators_and_rng():
    from smartsom.algorithms.rule_actions import RuleMetrics

    source = request("pre", "accept", [candidate("a"), candidate("STOP", ("STOP",))])
    projected, audit = project(source)
    metrics = RuleMetrics()
    metrics.add(projected, ("STOP",))
    rows = metrics.summary()["pre.accept"]
    assert rows["decisions"] == 1
    assert rows["physical_candidates"] == 2
    assert rows["projected_candidates"] == 2
    assert rows["rules"]["STOP"]["selection_rate"] == 1
    assert rows["rules"]["nearest"]["alias_rate"] == 1


def test_custom_codes_are_mapping_order_independent_and_do_not_wrap():
    names = tuple(f"custom_{i}" for i in range(8))
    source = request("dispatcher", "pickup", [candidate(name) for name in names])
    custom = {
        name: (lambda facts, decision, candidates, chosen=name: chosen)
        for name in names
    }
    first, _ = project(source, rules=names, custom=custom)
    second, _ = project(
        source, rules=names, custom=dict(reversed(tuple(custom.items())))
    )
    assert first == second
    assert len({c.features[45:] for c in first.candidates}) == len(names)
