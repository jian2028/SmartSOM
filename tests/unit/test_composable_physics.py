"""Hand boundaries and semantic replay of the v3 physical contract."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from smartsom import api
from smartsom.algorithms.pickup_matching import global_optimal, matching_rng
from smartsom.algorithms.production_composition import (
    BoundaryCoordinator,
    replay_boundary,
)
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.engine.production import ProductionSimulator

ROOT = Path(__file__).resolve().parents[2]


def example():
    return api.prepare(
        api.load_config(ROOT / "configs/test/runs/template1_warmup.yaml"),
        training=False,
    ).resolved.scenario


def coordinator(sim, matching="global_optimal"):
    names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": "nearest",
        "mover": "shortest_path",
    }
    return BoundaryCoordinator(
        sim,
        {r: RulePolicy(r, n) for r, n in names.items()},
        {r: {"default": r} for r in names},
        matching,
    )


def test_uniform_optimal_matching_includes_vehicle_subsets():
    rng = random.Random(51)
    draws = Counter(
        global_optimal(("j",), ("a", "b", "c"), lambda j, a: 0, rng)
        for _ in range(1200)
    )
    assert len(draws) == 3
    assert all(300 < value < 500 for value in draws.values())
    assert global_optimal(
        ("j1", "j2"), ("a", "b", "c"), lambda j, a: int(a == "c"), random.Random(1)
    ) in ((("a", "j1"), ("b", "j2")), (("b", "j1"), ("a", "j2")))


def test_matching_randomness_is_independent_and_identity_reproducible():
    args = (101, 2, "source", 11, ("j1", "j2"), ("a", "b", "c"))
    first = matching_rng(*args).getstate()
    random.seed(500)
    for _ in range(300):
        random.random()
    assert matching_rng(*args).getstate() == first


@pytest.mark.parametrize("matching", ["global_optimal", "priority_greedy"])
def test_rules_service_duration_and_replay(matching):
    scenario = example()
    sim = ProductionSimulator(scenario, contract="v3")
    driver = coordinator(sim, matching)
    replay = ProductionSimulator(scenario, contract="v3")
    saw_loading = False
    for _ in range(100):
        if sim.done:
            break
        row = driver.tick()
        assert row["tick"] == replay.tick + 1
        replay_boundary(replay, row)
        for agv, state in row["boundary_state"]["agvs"].items():
            if state["service"] and state["service"]["kind"] == "pickup":
                saw_loading = True
                job = state["service"]["job"]
                source = state["service"]["owner"]
                assert state["job"] is None
                assert row["boundary_state"]["jobs"][job]["location"] == source
                assert row["state"]["agvs"][agv]["job"] == job
                assert row["state"]["jobs"][job]["location"] == agv
        for record in row["decisions"]:
            if record["role"] == "machine":
                assert record["proposal"] is not None
            if record["role"] == "buffer":
                assert len(record["prefix"]) < record["count"]
    assert saw_loading


def test_transaction_restores_physics_on_invalid_proposal():
    sim = ProductionSimulator(example(), contract="v3")
    before = sim.snapshot()
    sim.protocol.begin()
    with pytest.raises(ValueError, match="invalid"):
        try:
            sim.protocol.accept_proposals({"missing": ("job", "normal")}, {})
        except ValueError:
            sim.protocol.abort()
            raise
    assert sim.snapshot() == before


def test_full_compatible_target_is_not_removed():
    sim = ProductionSimulator(example(), contract="v3")
    driver = coordinator(sim)
    while not sim.agvs[next(iter(sim.agvs))]["job"] and sim.tick < 20:
        driver.tick()
    vehicle = next(k for k, v in sim.agvs.items() if v["job"])
    job = sim.agvs[vehicle]["job"]
    owner = sim.protocol.destination_owners(job)[0]
    port = sim.protocol.ports_for(owner, "drop_off")[0]
    from copy import deepcopy

    for slot, capacity in sim.capacity[owner].items():
        assert capacity is not None
        while len(sim.storage[owner][slot]) < capacity:
            blocker = f"blocker:{slot}:{len(sim.storage[owner][slot])}"
            sim.jobs[blocker] = deepcopy(sim.jobs[job])
            sim.jobs[blocker].update(location="queue", slot=None)
            sim._place(blocker, owner, slot)
    sim.protocol.begin()
    targets = sim.protocol.dispatch_candidates(vehicle)
    assert any(
        c.action.owner == owner and c.action.port == port.port_id for c in targets
    )
    assert sim.protocol.drop_slot(owner, Counter(), port) is None
    slot = next(iter(sim.storage[owner]))
    sim._remove(sim.storage[owner][slot][0])
    assert sim.protocol.drop_slot(owner, Counter(), port) == slot
    sim.protocol.abort()


def _idle_proposals(sim, requests):
    return (
        {r.owner: r.candidates[0].action for r in requests if r.role == "machine"},
        {r.owner: r.candidates[0].action for r in requests if r.role == "dispatcher"},
    )


def _legal_movers(requests):
    return {r.owner: next(c.action for c in r.candidates if c.legal) for r in requests}


def test_two_pickup_ports_share_stock_and_load_parallel_with_one_tick():
    from smartsom.config.experiment_v4 import compile_experiment

    scenario = (
        compile_experiment(ROOT / "configs/test/runs/all_rules_v4.yaml")
        .entries[0]
        .prepared.scenario
    )
    sim = ProductionSimulator(scenario, contract="v3")
    source = next(b for b, role in sim.roles.items() if role == "system_input")
    ports = sim.protocol.ports_for(source, "pickup")
    assert len(ports) >= 2
    cars = list(sim.agvs)[:2]
    for car, port in zip(cars, ports[:2], strict=True):
        sim.agvs[car].update(
            cell=[port.cell.x, port.cell.y],
            target={"owner": source, "port": port.port_id},
            reservation=source,
            reservation_tick=0,
        )
    requests = sim.protocol.begin()
    machines, dispatch = _idle_proposals(sim, requests)
    buffers = sim.protocol.accept_proposals(machines, dispatch)
    request = next(r for r in buffers if r.owner == source)
    assert request.count == 2
    prefix = tuple(c.action for c in request.candidates[:2])
    stock = len(sim.protocol.ready(source))
    movers = sim.protocol.prepare_services(
        {source: prefix}, tuple(zip(cars, prefix, strict=True))
    )
    assert all(sim.jobs[j]["location"] == source for j in prefix)
    assert all(
        sim.agvs[c]["job"] is None and sim.agvs[c]["reservation"] is None for c in cars
    )
    assert sum(len(r) for r in sim.storage[source].values()) == stock
    result = sim.protocol.commit(_legal_movers(movers))
    assert result["tick"] == 1
    assert all(sim.agvs[c]["job"] == j for c, j in zip(cars, prefix, strict=True))
    assert sum(len(r) for r in sim.storage[source].values()) == stock - 2


def test_new_start_contributes_only_at_next_boundary_without_double_supply():
    sim = ProductionSimulator(example(), contract="v3")
    job = next(iter(sim.jobs))
    step = sim.demands[sim.jobs[job]["demand"]].steps[0]
    machine = next(
        m for m, r in sim.machines.items() if step.operation_type in r.operation_types
    )
    sim._remove(job)
    sim._place(job, sim.pre[machine], sim._free(sim.pre[machine]))
    post = sim.post[machine]
    requests = sim.protocol.begin()
    assert sim.protocol.supply[post] == 0
    machines, dispatch = _idle_proposals(sim, requests)
    assert machine in machines
    buffers = sim.protocol.accept_proposals(machines, dispatch)
    assert not buffers
    assert sim.protocol.supply[post] == 0
    movers = sim.protocol.prepare_services({}, ())
    sim.protocol.commit(_legal_movers(movers))
    sim.protocol.begin()
    assert sim.protocol.supply[post] == 1
    sim.protocol.abort()
    # Advance through completion without any source reservation.
    while sim.machine_state[machine]["status"] == "PROCESSING":
        requests = sim.protocol.begin()
        machines, dispatch = _idle_proposals(sim, requests)
        sim.protocol.accept_proposals(machines, dispatch)
        movers = sim.protocol.prepare_services({}, ())
        sim.protocol.commit(_legal_movers(movers))
    sim.protocol.begin()
    assert sim.protocol.supply[post] == 1
    assert sim.protocol.ready(post) == (job,)
    sim.protocol.abort()


def test_unserved_port_vehicle_must_clear_only_when_empty_nonport_exit_exists():
    sim = ProductionSimulator(example(), contract="v3")
    car = next(iter(sim.agvs))
    port = next(iter(sim.factory.ports))
    sim.agvs[car]["cell"] = [port.cell.x, port.cell.y]
    mask = sim.protocol.mover_mask(car)
    assert not mask[-1]
    from smartsom.domain.production import MOVES

    for allowed, (dx, dy) in zip(mask[:4], MOVES.values(), strict=True):
        if allowed:
            cell = port.cell.x + dx, port.cell.y + dy
            assert cell not in sim.ports
            assert cell not in {tuple(a["cell"]) for a in sim.agvs.values()}
    sim.agvs[car]["service"] = {"kind": "pickup"}
    assert sim.protocol.mover_mask(car) == (False, False, False, False, True)


def test_clearance_shortest_path_reserves_forced_port_exit():
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    observation = {
        "tick": 91,
        "topology": {
            "width": 12,
            "height": 12,
            "solids": [[8, 10]],
            "ports": {"port_020": [8, 9], "port_021": [9, 9]},
        },
        "agvs": {
            "agv_001": {
                "cell": [8, 8],
                "service": None,
                "target": {"owner": "buffer_009", "port": "port_020"},
            },
            "agv_002": {
                "cell": [8, 9],
                "service": None,
                "target": {"owner": "buffer_010", "port": "port_021"},
            },
            "agv_003": {
                "cell": [6, 9],
                "service": None,
                "target": {"owner": "buffer_009", "port": "port_020"},
            },
            "agv_004": {
                "cell": [7, 8],
                "service": None,
                "target": {"owner": "buffer_009", "port": "port_020"},
            },
        },
    }

    def choose(owner, legal):
        actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
        request = DecisionRequest(
            tick=91,
            stage="mover",
            role="mover",
            owner=owner,
            candidates=tuple(
                Candidate(action, action, (0.0,) * 12, action in legal)
                for action in actions
            ),
            observation=observation,
        )
        return RulePolicy("mover", "clearance_shortest_path").choose(request).action

    assert choose("agv_002", {"LEFT"}) == "LEFT"
    assert choose("agv_003", {"RIGHT", "WAIT"}) == "WAIT"
    assert choose("agv_004", {"DOWN", "WAIT"}) == "WAIT"


def test_clearance_shortest_path_allows_nonconflicting_moves_together():
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    observation = {
        "tick": 10,
        "topology": {
            "width": 12,
            "height": 12,
            "solids": [],
            "ports": {
                "target_1": [3, 1],
                "target_2": [3, 3],
            },
        },
        "agvs": {
            "agv_001": {
                "cell": [1, 1],
                "service": None,
                "target": {"owner": "one", "port": "target_1"},
            },
            "agv_002": {
                "cell": [1, 3],
                "service": None,
                "target": {"owner": "two", "port": "target_2"},
            },
        },
    }

    def choose(owner):
        actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
        request = DecisionRequest(
            tick=10,
            stage="mover",
            role="mover",
            owner=owner,
            candidates=tuple(
                Candidate(action, action, (0.0,) * 12) for action in actions
            ),
            observation=observation,
        )
        return RulePolicy("mover", "clearance_shortest_path").choose(request).action

    assert choose("agv_001") == "RIGHT"
    assert choose("agv_002") == "RIGHT"


def test_clearance_shortest_path_does_not_reenter_an_unrelated_port():
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    observation = {
        "tick": 126,
        "topology": {
            "width": 6,
            "height": 10,
            "solids": [[3, 6]],
            "ports": {"target": [1, 6], "unrelated": [3, 7]},
        },
        "agvs": {
            "agv_001": {
                "cell": [3, 8],
                "service": None,
                "target": {"owner": "input", "port": "target"},
            }
        },
    }
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
    request = DecisionRequest(
        tick=126,
        stage="mover",
        role="mover",
        owner="agv_001",
        candidates=tuple(Candidate(action, action, (0.0,) * 12) for action in actions),
        observation=observation,
    )

    action = RulePolicy("mover", "clearance_shortest_path").choose(request).action

    assert action == "LEFT"


def test_clearance_shortest_path_stages_until_pickup_is_ready():
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    observation = {
        "tick": 222,
        "topology": {
            "width": 6,
            "height": 6,
            "solids": [],
            "ports": {"target": [3, 2]},
        },
        "sources": {"buffer": {"ready": [], "supply": 1, "reserved": 1}},
        "agvs": {
            "agv_001": {
                "cell": [3, 3],
                "job": None,
                "reservation": "buffer",
                "service": None,
                "target": {"owner": "buffer", "port": "target"},
            }
        },
    }
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
    policy = RulePolicy("mover", "clearance_shortest_path")

    def choose():
        request = DecisionRequest(
            tick=observation["tick"],
            stage="mover",
            role="mover",
            owner="agv_001",
            candidates=tuple(
                Candidate(action, action, (0.0,) * 12) for action in actions
            ),
            observation=observation,
        )
        return policy.choose(request).action

    assert choose() == "WAIT"
    waiting_key = policy._mover_plan_key
    observation["sources"]["buffer"]["ready"] = ["order_01/attempt/1"]
    assert choose() == "UP"
    assert policy._mover_plan_key != waiting_key


def test_clearance_shortest_path_does_not_reverse_for_the_same_task():
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    observation = {
        "tick": 1,
        "topology": {
            "width": 6,
            "height": 6,
            "solids": [],
            "ports": {"target": [3, 2]},
        },
        "agvs": {
            "agv_001": {
                "cell": [1, 1],
                "job": "order_01/attempt/1",
                "service": None,
                "target": {"owner": "buffer", "port": "target"},
            },
            "agv_002": {
                "cell": [4, 4],
                "job": None,
                "service": None,
                "target": {"owner": "buffer", "port": "target"},
            },
        },
    }
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
    policy = RulePolicy("mover", "clearance_shortest_path")

    def choose():
        request = DecisionRequest(
            tick=observation["tick"],
            stage="mover",
            role="mover",
            owner="agv_001",
            candidates=tuple(
                Candidate(action, action, (0.0,) * 12) for action in actions
            ),
            observation=observation,
        )
        return policy.choose(request).action

    assert choose() == "DOWN"
    saved = json.loads(json.dumps(policy.state_dict()))
    policy = RulePolicy("mover", "clearance_shortest_path")
    policy.load_state_dict(saved)
    observation["tick"] = 2
    observation["agvs"]["agv_001"]["cell"] = [1, 2]
    observation["agvs"]["agv_002"].update(cell=[2, 2], target=None)
    assert choose() != "UP"


def test_clearance_shortest_path_routes_around_an_idle_agv():
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    observation = {
        "tick": 232,
        "topology": {
            "width": 6,
            "height": 6,
            "solids": [],
            "ports": {"target": [3, 2]},
        },
        "agvs": {
            "agv_001": {
                "cell": [2, 4],
                "service": None,
                "target": {"owner": "buffer", "port": "target"},
            },
            "agv_002": {"cell": [2, 3], "service": None, "target": None},
        },
    }
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
    request = DecisionRequest(
        tick=232,
        stage="mover",
        role="mover",
        owner="agv_001",
        candidates=tuple(Candidate(action, action, (0.0,) * 12) for action in actions),
        observation=observation,
    )

    action = RulePolicy("mover", "clearance_shortest_path").choose(request).action

    assert action == "RIGHT"


def _large_fleet_observation(count=100):
    return {
        "tick": 17,
        "topology": {
            "width": count * 3 + 1,
            "height": 3,
            "solids": [],
            "ports": {f"target_{i:03d}": [i * 3 + 1, 0] for i in range(count)},
        },
        "agvs": {
            f"agv_{i:03d}": {
                "cell": [i * 3 + 1, 1],
                "job": f"job_{i:03d}",
                "service": None,
                "target": {"owner": f"output_{i:03d}", "port": f"target_{i:03d}"},
            }
            for i in range(count)
        },
    }


def _mover_request(observation, owner):
    from smartsom.domain.production_decisions import Candidate, DecisionRequest

    actions = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
    return DecisionRequest(
        tick=observation["tick"],
        stage="mover",
        role="mover",
        owner=owner,
        candidates=tuple(Candidate(action, action, (0.0,) * 12) for action in actions),
        observation=observation,
    )


def test_clearance_shortest_path_scales_to_one_hundred_agvs_and_reuses_plan():
    from smartsom.domain.production import MOVES

    observation = _large_fleet_observation()
    policy = RulePolicy("mover", "clearance_shortest_path")
    owners = sorted(observation["agvs"], reverse=True)
    actions = {
        owner: policy.choose(_mover_request(observation, owner)).action
        for owner in owners
    }
    cached = policy._mover_plan
    assert actions == {owner: "UP" for owner in owners}
    assert all(
        policy.choose(_mover_request(observation, owner)).action == actions[owner]
        for owner in reversed(owners)
    )
    assert policy._mover_plan is cached
    destinations = {
        owner: (
            observation["agvs"][owner]["cell"][0] + MOVES.get(action, (0, 0))[0],
            observation["agvs"][owner]["cell"][1] + MOVES.get(action, (0, 0))[1],
        )
        for owner, action in actions.items()
    }
    assert len(set(destinations.values())) == len(owners)


def test_clearance_shortest_path_allows_a_following_chain_without_swaps():
    from smartsom.domain.production import MOVES

    observation = {
        "tick": 0,
        "topology": {
            "width": 7,
            "height": 3,
            "solids": [],
            "ports": {"target": [6, 1]},
        },
        "agvs": {
            f"agv_{i}": {
                "cell": [i, 1],
                "job": f"job_{i}",
                "service": None,
                "target": {"owner": "output", "port": "target"},
            }
            for i in (1, 2, 3)
        },
    }
    policy = RulePolicy("mover", "clearance_shortest_path")
    actions = {
        owner: policy.choose(_mover_request(observation, owner)).action
        for owner in observation["agvs"]
    }
    assert set(actions.values()) == {"RIGHT"}
    destinations = {
        owner: tuple(
            cell + delta
            for cell, delta in zip(
                observation["agvs"][owner]["cell"],
                MOVES[actions[owner]],
                strict=True,
            )
        )
        for owner in actions
    }
    assert len(set(destinations.values())) == len(actions)


def test_clearance_shortest_path_moves_an_idle_vehicle_off_a_port():
    observation = {
        "tick": 4,
        "topology": {
            "width": 3,
            "height": 3,
            "solids": [[1, 0], [1, 2], [2, 1]],
            "ports": {"occupied": [1, 1]},
        },
        "agvs": {
            "agv_idle": {
                "cell": [1, 1],
                "job": None,
                "service": None,
                "target": None,
            }
        },
    }
    request = _mover_request(observation, "agv_idle")
    restricted = request.__class__(
        request.tick,
        request.stage,
        request.role,
        request.owner,
        tuple(
            candidate.__class__(
                candidate.identity,
                candidate.action,
                candidate.features,
                candidate.action == "LEFT",
            )
            for candidate in request.candidates
        ),
        request.observation,
    )
    assert (
        RulePolicy("mover", "clearance_shortest_path").choose(restricted).action
        == "LEFT"
    )


def test_idle_vehicle_clears_a_port_as_its_occupied_exit_becomes_free():
    observation = {
        "tick": 4,
        "topology": {
            "width": 4,
            "height": 3,
            "solids": [[2, 0], [2, 2], [3, 1]],
            "ports": {"occupied": [2, 1], "output": [1, 0]},
        },
        "agvs": {
            "idle": {"cell": [2, 1], "target": None, "service": None, "job": None},
            "loaded": {
                "cell": [1, 1],
                "target": {"owner": "output", "port": "output"},
                "service": None,
                "job": "job",
            },
        },
    }
    policy = RulePolicy("mover", "clearance_shortest_path")
    assert policy.choose(_mover_request(observation, "loaded")).action == "UP"
    assert policy.choose(_mover_request(observation, "idle")).action == "LEFT"


def test_transit_port_exit_is_cleared_before_a_loaded_vehicle_enters():
    observation = {
        "tick": 4,
        "topology": {
            "width": 5,
            "height": 3,
            "solids": [[2, 0], [2, 2]],
            "ports": {"transit": [2, 1], "output": [4, 1]},
        },
        "agvs": {
            "idle": {"cell": [3, 1], "target": None, "service": None, "job": None},
            "loaded": {
                "cell": [1, 1],
                "target": {"owner": "output", "port": "output"},
                "service": None,
                "job": "job",
            },
        },
    }
    policy = RulePolicy("mover", "clearance_shortest_path")
    assert policy.choose(_mover_request(observation, "idle")).action == "UP"
    observation["tick"] += 1
    observation["agvs"]["idle"]["cell"] = [3, 0]
    observation["agvs"]["loaded"]["cell"] = [2, 1]
    assert policy.choose(_mover_request(observation, "loaded")).action == "RIGHT"


def test_inventory_shortage_limits_prefix_not_in_transit_cars():
    sim = ProductionSimulator(example(), contract="v3")
    source = next(b for b, role in sim.roles.items() if role == "system_input")
    jobs = list(sim.protocol.ready(source))
    for job in jobs[1:]:
        sim._remove(job)
        sim.jobs[job]["location"] = "queue"
    ports = sim.protocol.ports_for(source, "pickup")
    cars = list(sim.agvs)
    for car in cars:
        sim.agvs[car].update(
            target={"owner": source, "port": ports[0].port_id},
            reservation=source,
            reservation_tick=0,
        )
    for car, port in zip(cars[:2], ports[:2], strict=True):
        sim.agvs[car]["cell"] = [port.cell.x, port.cell.y]
        sim.agvs[car]["target"]["port"] = port.port_id
    requests = sim.protocol.begin()
    machines, dispatch = _idle_proposals(sim, requests)
    buffers = sim.protocol.accept_proposals(machines, dispatch)
    assert buffers[0].count == 1
    assert set(sim.protocol.service_vehicles[source]) == set(cars[:2])
    sim.protocol.abort()
