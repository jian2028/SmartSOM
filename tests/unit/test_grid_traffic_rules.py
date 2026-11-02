"""Grid traffic rule variants: yielding movement and sticky, spread dispatch."""

from collections import Counter

import pytest

from smartsom.algorithms import production_rules
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.domain.production import MOVES
from smartsom.domain.production_decisions import (
    Candidate,
    DecisionRequest,
    DispatchTarget,
)

# A 5x3 map whose middle row is a corridor; row 0 and row 2 are open lanes.
TOPOLOGY = {
    "width": 5,
    "height": 3,
    "solids": [],
    "ports": {"west": [0, 1], "east": [4, 1]},
}


def vehicle(cell, port=None, job=None, owner=None):
    target = {"owner": owner or port, "port": port} if port else None
    return {"cell": list(cell), "target": target, "job": job, "service": None}


def mover_request(owner, agvs, *, tick=0, wait_legal=True, solids=(), ready=True):
    topology = {**TOPOLOGY, "solids": [list(c) for c in solids]}
    x, y = agvs[owner]["cell"]
    candidates = []
    for action, (dx, dy) in MOVES.items():
        nxt = x + dx, y + dy
        legal = 0 <= nxt[0] < 5 and 0 <= nxt[1] < 3 and nxt not in set(solids)
        candidates.append(Candidate(action, action, (), legal))
    candidates.append(Candidate("WAIT", "WAIT", (), wait_legal))
    sources = {"east": {"ready": ["job"] if ready else [], "supply": 0, "reserved": 0}}
    view = {"tick": tick, "agvs": agvs, "topology": topology, "sources": sources}
    return DecisionRequest(tick, "mover", "mover", owner, tuple(candidates), view)


def choose(policy, request):
    return policy.choose(request).action


def test_yield_mover_follows_the_free_shortest_path():
    agvs = {"a": vehicle((1, 1), "east")}
    policy = RulePolicy("mover", "shortest_path_yield", seed=1)
    assert choose(policy, mover_request("a", agvs)) == "RIGHT"


def test_blocked_yield_mover_waits_instead_of_detouring(monkeypatch):
    # b blocks a's only shortest step; shortest_path detours around it.
    agvs = {"a": vehicle((1, 1), "east"), "b": vehicle((2, 1), "west")}
    request = mover_request("a", agvs)
    assert choose(RulePolicy("mover", "shortest_path", seed=1), request) in {
        "UP",
        "DOWN",
    }
    monkeypatch.setattr(production_rules, "SIDESTEP_PROBABILITY", 0.0)
    assert choose(RulePolicy("mover", "shortest_path_yield", seed=1), request) == "WAIT"


def test_blocked_yield_mover_sometimes_steps_aside_into_free_cells_only():
    agvs = {"a": vehicle((1, 1), "east"), "b": vehicle((2, 1), "west")}
    request = mover_request("a", agvs)
    policy = RulePolicy("mover", "shortest_path_yield", seed=3)
    actions = Counter(choose(policy, request) for _ in range(400))
    assert set(actions) <= {"WAIT", "UP", "DOWN", "LEFT"}
    assert actions["RIGHT"] == 0
    assert 0.6 < actions["WAIT"] / 400 < 0.9
    assert actions["UP"] + actions["DOWN"] + actions["LEFT"] > 0


def test_yield_mover_must_leave_when_waiting_is_illegal():
    agvs = {"a": vehicle((1, 1), "east"), "b": vehicle((2, 1), "west")}
    request = mover_request("a", agvs, wait_legal=False)
    policy = RulePolicy("mover", "shortest_path_yield", seed=1)
    assert choose(policy, request) in {"UP", "DOWN", "LEFT"}


def test_yield_mover_is_reproducible_for_a_seed():
    agvs = {"a": vehicle((1, 1), "east"), "b": vehicle((2, 1), "west")}
    request = mover_request("a", agvs)

    def actions():
        policy = RulePolicy("mover", "shortest_path_yield", seed=9)
        return [choose(policy, request) for _ in range(50)]

    assert actions() == actions()


def test_idle_empty_vehicle_parks_and_only_gives_way_when_crowded():
    alone = {"a": vehicle((1, 1), "east")}
    policy = RulePolicy("mover", "shortest_path_yield", seed=4)
    request = mover_request("a", alone, ready=False)
    assert {choose(policy, request) for _ in range(50)} == {"WAIT"}
    loaded = {"a": vehicle((1, 1), "east", job="job")}
    assert choose(policy, mover_request("a", loaded, ready=False)) == "RIGHT"
    crowded = {"a": vehicle((1, 1), "east"), "b": vehicle((1, 0), "west")}
    actions = Counter(
        choose(policy, mover_request("a", crowded, ready=False)) for _ in range(400)
    )
    assert actions["UP"] == 0 and 0.6 < actions["WAIT"] / 400 < 0.9


PORTS = {"near": [1, 0], "far": [4, 0], "other": [3, 2]}


def dispatcher_request(owner, agvs, targets):
    candidates = tuple(
        Candidate(
            f"TARGET:{port}:{port}",
            DispatchTarget(port, port),
            (1.0, distance, 0, 0, 0, 1.0),
            True,
        )
        for port, distance in targets
    )
    view = {
        "tick": 0,
        "agvs": agvs,
        "topology": {**TOPOLOGY, "ports": PORTS},
        "sources": {},
    }
    return DecisionRequest(0, "dispatcher", "dispatcher", owner, candidates, view)


def target(policy, request):
    return policy.choose(request).action.port


@pytest.mark.parametrize("job", [None, "job"])
def test_sticky_dispatcher_keeps_an_unreached_target(job):
    agvs = {"a": vehicle((0, 0), "far", job=job)}
    request = dispatcher_request("a", agvs, [("near", 0.01), ("far", 0.04)])
    assert target(RulePolicy("dispatcher", "nearest", seed=1), request) == "near"
    assert target(RulePolicy("dispatcher", "nearest_sticky", seed=1), request) == "far"


def test_sticky_dispatcher_repicks_after_reaching_its_target():
    agvs = {"a": vehicle(PORTS["far"], "far")}
    request = dispatcher_request("a", agvs, [("near", 0.01), ("far", 0.04)])
    assert target(RulePolicy("dispatcher", "nearest_sticky", seed=1), request) == "near"


def test_sticky_dispatcher_spreads_empty_vehicles_over_unclaimed_targets():
    agvs = {"a": vehicle((0, 0)), "b": vehicle((2, 2), "near")}
    request = dispatcher_request("a", agvs, [("near", 0.01), ("other", 0.03)])
    assert target(RulePolicy("dispatcher", "nearest", seed=1), request) == "near"
    assert (
        target(RulePolicy("dispatcher", "nearest_sticky", seed=1), request) == "other"
    )
    # With every target claimed, it still chooses the nearest one.
    agvs["c"] = vehicle((4, 2), "other")
    assert target(RulePolicy("dispatcher", "nearest_sticky", seed=1), request) == "near"


def park_request(owner, agvs, *, east_ready=False, east_supply=1, **options):
    # Another source holds a job, so the vehicle is not globally idle; the
    # target (east) is pending while its machine still processes (supply 1).
    request = mover_request(owner, agvs, **options)
    request.observation["sources"] = {
        "east": {
            "ready": ["job"] if east_ready else [],
            "supply": east_supply,
            "reserved": 0,
        },
        "west": {"ready": ["other"], "supply": 1, "reserved": 0},
    }
    return request


def parking_policy(seed=1):
    return RulePolicy(
        "mover",
        "shortest_path_yield",
        seed=seed,
        parameters={"park_while_target_pending": True},
    )


def test_parking_mover_waits_beside_a_pending_target_port():
    agvs = {"a": vehicle((3, 1), "east")}
    assert choose(parking_policy(), park_request("a", agvs)) == "WAIT"
    # The default mover still steps onto the empty port.
    default = RulePolicy("mover", "shortest_path_yield", seed=1)
    assert choose(default, park_request("a", agvs)) == "RIGHT"


def test_parking_mover_enters_once_the_target_has_a_job():
    agvs = {"a": vehicle((3, 1), "east")}
    request = park_request("a", agvs, east_ready=True)
    assert choose(parking_policy(), request) == "RIGHT"


def test_parking_mover_still_reaches_a_target_with_nothing_coming():
    # Arriving at an empty source is what triggers a fresh dispatch decision.
    agvs = {"a": vehicle((3, 1), "east")}
    request = park_request("a", agvs, east_supply=0)
    assert choose(parking_policy(), request) == "RIGHT"


def test_parking_mover_approaches_normally_from_afar():
    agvs = {"a": vehicle((1, 1), "east")}
    assert choose(parking_policy(), park_request("a", agvs)) == "RIGHT"


def test_parking_mover_forced_off_the_port_leaves_to_a_non_port_cell():
    agvs = {"a": vehicle((4, 1), "east")}
    request = park_request("a", agvs, wait_legal=False)
    actions = {choose(parking_policy(seed), request) for seed in range(20)}
    assert actions <= {"LEFT", "UP", "DOWN"}


def test_parked_mover_ignores_neighbours_waiting_for_the_same_port():
    agvs = {"a": vehicle((3, 1), "east"), "b": vehicle((3, 0), "east")}
    policy = parking_policy(seed=5)
    assert {choose(policy, park_request("a", agvs)) for _ in range(100)} == {"WAIT"}


def test_parked_mover_sometimes_gives_way_to_a_loaded_neighbour():
    agvs = {"a": vehicle((3, 1), "east"), "b": vehicle((2, 1), "west", job="j")}
    policy = parking_policy(seed=5)
    actions = Counter(choose(policy, park_request("a", agvs)) for _ in range(400))
    assert actions["RIGHT"] == 0 and actions["LEFT"] == 0
    assert 0.6 < actions["WAIT"] / 400 < 0.9


@pytest.mark.parametrize(
    "parameters",
    [{"park_while_target_pending": "yes"}, {"park": True}],
)
def test_parking_parameter_is_validated(parameters):
    with pytest.raises(ValueError):
        RulePolicy("mover", "shortest_path_yield", seed=1, parameters=parameters)
