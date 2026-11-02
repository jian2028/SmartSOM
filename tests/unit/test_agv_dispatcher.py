"""Public fleet admission and destination routing regressions."""

from copy import deepcopy

from smartsom.algorithms.agv_dispatcher import destination_score, pickup_plan
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.domain.production_decisions import (
    Candidate,
    DecisionRequest,
    DispatchTarget,
)


def view():
    return {
        "tick": 1,
        "topology": {
            "width": 12,
            "height": 6,
            "solids": [],
            "ports": {"input_port": [1, 0], "post_port": [9, 0]},
        },
        "sources": {
            "input": {"ready": ["new"], "supply": 1, "reserved": 0},
            "post": {"ready": ["wip"], "supply": 1, "reserved": 0},
        },
        "jobs": {"new": {"step": 0}, "wip": {"step": 3}},
        "agvs": {
            f"agv_{i}": {
                "cell": [x, 1],
                "job": None,
                "target": None,
                "service": None,
                "reservation": None,
            }
            for i, x in enumerate((8, 2, 5, 10))
        },
    }


def targets():
    return [DispatchTarget("input", "input_port"), DispatchTarget("post", "post_port")]


def test_admission_assigns_nearest_vehicle_once_per_ready_source():
    observation = view()
    observation["sources"]["input"].update(ready=["new"] * 10, supply=10)
    plan = pickup_plan(observation, targets(), {"work_in_progress_first": True})
    assert plan == {"agv_0": targets()[1], "agv_1": targets()[0]}


def test_admission_respects_ready_inventory_and_existing_reservations():
    observation = view()
    observation["sources"]["post"].update(ready=[], supply=2)
    observation["sources"]["input"]["reserved"] = 1
    assert not pickup_plan(observation, targets(), {})


def test_wip_priority_and_fleet_budget_include_loaded_vehicles():
    observation = view()
    observation["agvs"]["agv_3"]["job"] = "loaded"
    assert pickup_plan(observation, targets(), {"work_in_progress_first": True}) == {
        "agv_0": targets()[1]
    }


def test_shared_dispatch_is_independent_of_request_order_and_restoration():
    observation = view()
    candidates = (Candidate("NO_REQUEST", None, (0.0,) * 12),) + tuple(
        Candidate(f"TARGET:{target.owner}:{target.port}", target, (0.0,) * 12)
        for target in targets()
    )

    def choose(policy, owner):
        return policy.choose(
            DecisionRequest(
                1, "proposals", "dispatcher", owner, candidates, deepcopy(observation)
            )
        ).action

    params = {"fleet_admission": "traffic", "work_in_progress_first": True}
    policy = RulePolicy("dispatcher", "nearest", parameters=params)
    first = {owner: choose(policy, owner) for owner in observation["agvs"]}
    saved = policy.state_dict()
    policy.load_state_dict(saved)
    second = {owner: choose(policy, owner) for owner in reversed(observation["agvs"])}
    assert (
        first
        == second
        == {"agv_0": targets()[1], "agv_1": targets()[0], "agv_2": None, "agv_3": None}
    )


def test_full_destination_does_not_win_over_available_destination():
    observation = view()
    features = [0.0] * 12
    features[11] = 1.0
    observation["machines"] = {"busy": {"job": "other"}}
    current = Candidate("busy", DispatchTarget("busy", "old"), tuple(features))
    free = Candidate("free", DispatchTarget("free", "new"), (0.0,) * 12)
    assert destination_score(observation, "agv_0", free) < destination_score(
        observation, "agv_0", current
    )


def test_static_distance_cache_is_invalidated_for_changed_map():
    observation = view()
    cache = {}
    expected = pickup_plan(observation, targets(), {}, distance_cache=cache)
    assert pickup_plan(observation, targets(), {}, distance_cache=cache) == expected
    observation["topology"]["solids"] = [[x, 1] for x in range(12)]
    assert not pickup_plan(observation, targets(), {}, distance_cache=cache)


def two_sided_view():
    observation = view()
    observation["topology"]["ports"] = {"upper": [2, 0], "lower": [2, 5]}
    observation["sources"] = {"station": {"ready": ["a", "b", "c"], "reserved": 0}}
    observation["agvs"]["agv_0"]["cell"] = [2, 1]
    observation["agvs"]["agv_1"]["cell"] = [2, 4]
    return observation, [DispatchTarget("station", side) for side in ("upper", "lower")]


def test_two_sides_collect_different_ready_jobs_concurrently():
    observation, choices = two_sided_view()
    plan = pickup_plan(observation, choices, {"fleet_admission": "traffic"})
    assert plan == {"agv_0": choices[0], "agv_1": choices[1]}
    assert len(set(plan.values())) == len(plan)


def test_busy_side_does_not_block_opposite_side():
    observation, choices = two_sided_view()
    observation["agvs"]["agv_0"].update(
        target={"owner": "station", "port": "upper"}, reservation="station"
    )
    observation["sources"]["station"]["reserved"] = 1
    assert pickup_plan(observation, choices, {"fleet_admission": "traffic"}) == {
        "agv_1": choices[1]
    }


def test_two_sides_cannot_overclaim_one_ready_job():
    observation, choices = two_sided_view()
    observation["sources"]["station"]["ready"] = ["a"]
    assert len(pickup_plan(observation, choices, {"fleet_admission": "traffic"})) == 1


def test_traffic_admission_is_not_limited_by_square_root_of_fleet():
    observation = view()
    observation["topology"]["ports"] = {f"p{i}": [i * 3, 0] for i in range(4)}
    observation["sources"] = {f"s{i}": {"ready": [f"j{i}"]} for i in range(4)}
    choices = [DispatchTarget(f"s{i}", f"p{i}") for i in range(4)]
    assert len(pickup_plan(observation, choices, {"fleet_admission": "traffic"})) == 4
    assert (
        len(
            pickup_plan(
                observation, choices, {"fleet_admission": "traffic", "max_active": 2}
            )
        )
        == 2
    )


def test_available_current_destination_is_committed_despite_port_competition():
    observation = view()
    observation["agvs"]["agv_1"]["target"] = {"owner": "post", "port": "post_port"}
    features = [0.0] * 12
    features[11] = 1.0
    current = Candidate("current", DispatchTarget("post", "post_port"), tuple(features))
    other = Candidate("other", DispatchTarget("input", "input_port"), (0.0,) * 12)
    assert destination_score(observation, "agv_0", current) < destination_score(
        observation, "agv_0", other
    )


def test_target_only_request_does_not_lose_work_to_an_uncalled_closer_vehicle():
    observation, choices = two_sided_view()
    observation["sources"]["station"]["reserved"] = 99  # nonexclusive intentions
    observation["topology"]["ports"]["empty"] = [11, 5]
    observation["sources"]["empty"] = {"ready": [], "supply": 0}
    actions = [*choices, DispatchTarget("empty", "empty")]
    request = DecisionRequest(
        1,
        "proposals",
        "dispatcher",
        "agv_3",
        tuple(Candidate(str(action), action, (0.0,) * 12) for action in actions),
        observation,
    )
    policy = RulePolicy(
        "dispatcher", "nearest", parameters={"fleet_admission": "traffic"}
    )
    assert policy.choose(request).action.owner == "station"


def test_current_intentions_use_the_unqueued_opposite_station_side():
    observation, choices = two_sided_view()
    observation["agvs"]["agv_0"]["target"] = {"owner": "station", "port": "upper"}
    observation["sources"]["station"]["reserved"] = 99
    request = DecisionRequest(
        1,
        "proposals",
        "dispatcher",
        "agv_1",
        tuple(Candidate(str(action), action, (0.0,) * 12) for action in choices),
        observation,
    )
    policy = RulePolicy(
        "dispatcher", "nearest", parameters={"fleet_admission": "traffic"}
    )
    assert policy.choose(request).action == choices[1]


def test_final_unknown_job_prefers_inspection_before_blind_delivery():
    observation = view()
    observation["agvs"]["agv_0"]["job"] = "finished"
    observation["jobs"]["finished"] = {"quality": "UNKNOWN", "next_operation": None}
    observation["stations"] = {"inspection": {}}
    inspection = Candidate(
        "inspection", DispatchTarget("inspection", "post_port"), (0.0,) * 12
    )
    output = Candidate("output", DispatchTarget("output", "input_port"), (0.0,) * 12)
    assert destination_score(observation, "agv_0", inspection) < destination_score(
        observation, "agv_0", output
    )
    observation["jobs"]["finished"]["quality"] = "PASS"
    assert (
        destination_score(observation, "agv_0", inspection)[0]
        == destination_score(observation, "agv_0", output)[0]
    )
