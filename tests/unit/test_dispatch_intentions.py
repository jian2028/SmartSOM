"""Persistent intention, empty episode, geometry progress and FIFO invariants."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_dispatcher_targets import driver, scenario

from smartsom.algorithms.pickup_matching import first_arrival, matching_rng
from smartsom.algorithms.production_composition import replay_boundary
from smartsom.algorithms.production_rules import PolicyChoice, RulePolicy
from smartsom.domain.production_decisions import (
    BoundaryCommand,
    Candidate,
    DecisionRequest,
    DispatchTarget,
)
from smartsom.engine.production import ProductionSimulator


def fixed_empty_source(manual=False):
    original = scenario(True)
    if manual:
        first = original.factory.agvs[0].agv_id
        matrix = replace(
            original.transport_matrix,
            source="manual",
            times=tuple(
                (
                    a,
                    b,
                    0
                    if a == b
                    else 1
                    if a == "initial:" + first
                    else 6
                    if a.startswith("initial:")
                    else 4,
                )
                for a, b, _ in original.transport_matrix.times
            ),
        )
        original = replace(original, transport_matrix=matrix)
    sim = ProductionSimulator(original, contract="v3")
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    controller = driver(sim)
    controller.policies["dispatcher"].choose = lambda request: PolicyChoice(
        DispatchTarget(owner, port.port_id)
    )
    return sim, controller, owner, port


def test_nonexclusive_empty_intentions_keep_enroute_event_and_arrived_reconsideration():
    sim, controller, owner, _ = fixed_empty_source()
    first = controller.tick()
    assert len(first["actions"]["dispatchers"]) == len(sim.agvs)
    assert all(a["target"]["owner"] == owner for a in sim.agvs.values())
    assert all(a["reservation"] is None for a in sim.agvs.values())
    assert sim.protocol.reserved(owner) == len(sim.agvs)
    assert not first["rejections"]
    while sim.tick < 50:
        eligible = {
            v
            for v, a in sim.agvs.items()
            if a["travel"] is None or not a["empty_notified"]
        }
        row = controller.tick()
        assert {v for v, _ in row["actions"]["dispatchers"]} == eligible
    assert all(a["travel"] is None for a in sim.agvs.values())
    assert len(controller.tick()["actions"]["dispatchers"]) == len(sim.agvs)


def test_same_target_preserves_trip_and_progress_from_actual_cell():
    sim, controller, _, port = fixed_empty_source()
    controller.tick()
    vehicle = next(v for v, a in sim.agvs.items() if a["travel"])
    before = deepcopy(sim.agvs[vehicle]["travel"])
    actual = tuple(sim.agvs[vehicle]["cell"])
    assert actual == tuple(before["path"][0])
    controller.tick()
    after = sim.agvs[vehicle]["travel"]
    if after:
        assert after["departed_at"] == before["departed_at"]
        assert after["arrival_tick"] == before["arrival_tick"]
        assert after["path"] == before["path"]
    # Direct departure to another endpoint must start at current, not initial.
    state = sim.agvs[vehicle]
    current = tuple(state["cell"])
    other = next(
        p
        for p in sim.protocol.ports.values()
        if p.port_id != port.port_id
        and sim.protocol.distance(current, (p.cell.x, p.cell.y)) > 1
    )
    sim.protocol.depart(vehicle, other)
    travel = state["travel"]
    assert travel["departed_at"] == sim.tick and travel["progress"] == 0
    assert len(travel["path"]) == sim.protocol.distance(
        current, (other.cell.x, other.cell.y)
    )
    previous = current
    for cell in travel["path"]:
        assert cell not in sim.solids
        assert abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]) == 1
        previous = cell


def test_work_rearms_empty_episode_but_future_demand_does_not():
    sim, controller, owner, _ = fixed_empty_source()
    controller.tick()
    while any(a["travel"] for a in sim.agvs.values()):
        controller.tick()
    machine = next(m for m, post in sim.post.items() if post == owner)
    job = next(iter(sim.jobs))
    sim._remove(job)
    slot = next(iter(sim.storage[sim.pre[machine]]))
    sim._place(job, sim.pre[machine], slot)
    requests = sim.protocol.begin()
    assert not any(r.role == "dispatcher" for r in requests)
    assert all(not a["empty_notified"] for a in sim.agvs.values())
    # Close the test-only boundary, preserving the observed latch reset.
    sim.protocol.stage, sim.protocol.backup = "closed", None
    sim._remove(job)
    sim._place(job, sim._input(sim.demands[sim.jobs[job]["demand"]]), "pool")
    requests = sim.protocol.begin()
    assert len([r for r in requests if r.role == "dispatcher"]) == len(sim.agvs)
    sim.protocol.abort()


def test_abort_restores_consumed_empty_episode():
    sim, controller, _, _ = fixed_empty_source()
    controller.tick()
    while any(a["travel"] for a in sim.agvs.values()):
        controller.tick()
    before = sim.snapshot()
    requests = sim.protocol.begin()
    assert any(r.role == "dispatcher" for r in requests)
    assert all(a["empty_notified"] for a in sim.agvs.values())
    sim.protocol.abort()
    assert sim.snapshot() == before


@pytest.mark.parametrize("manual", [False, True])
def test_last_pickup_before_late_arrival_reconsiders_after_arrival(manual):
    sim, controller, owner, port = fixed_empty_source(manual)
    job = next(iter(sim.jobs))
    sim._remove(job)
    sim.jobs[job].update(
        step=len(sim.demands[sim.jobs[job]["demand"]].steps), quality="PASS"
    )
    sim._place(job, owner, next(iter(sim.storage[owner])))
    controller.policies["dispatcher"].choose = lambda request: PolicyChoice(
        DispatchTarget(owner, port.port_id)
        if request.observation["agvs"][request.owner]["job"] is None
        else request.candidates[0].action
    )
    distances = {v: sim.protocol.travel_cost(v, port) for v in sim.agvs}
    late = max(distances, key=distances.get)
    assert distances[late] > min(distances.values())
    decisions = []
    while sim.tick <= distances[late] + 2:
        row = controller.tick()
        decisions.extend(
            d["tick"]
            for d in row["decisions"]
            if d["role"] == "dispatcher" and d["owner"] == late
        )
    assert sim.agvs[late]["job"] is None and sim.agvs[late]["travel"] is None
    assert decisions[0] == 0 and len(decisions) >= 3
    assert decisions[1] < distances[late]
    assert distances[late] in decisions
    assert sim.agvs[late]["empty_notified"]
    assert sim.metrics["pickup_services"] == 1


def test_grid_actual_arrivals_have_physical_timestamps():
    sim = ProductionSimulator(scenario(False), contract="v3")
    controller = driver(sim)
    arrivals = []
    for _ in range(40):
        row = controller.tick()
        for vehicle, state in sim.agvs.items():
            if not state["target"]:
                continue
            port = sim.protocol.ports[state["target"]["port"]]
            before = row["boundary_state"]["agvs"][vehicle]
            if sim.protocol.arrived(state, port) and not sim.protocol.arrived(
                before, port
            ):
                assert state["arrived_at"] == sim.tick
                assert row["state"]["agvs"][vehicle]["arrived_at"] == sim.tick
                arrivals.append(sim.tick)
    assert len(set(arrivals)) > 1


def test_source_fifo_preserves_buffer_order_and_seeded_ties():
    jobs, vehicles = ("j1", "j2"), ("later", "early", "tie")
    arrival = {"early": 1, "tie": 1, "later": 2}

    def sample(seed):
        return first_arrival(
            jobs,
            vehicles,
            arrival.__getitem__,
            matching_rng(seed, 0, "source", 4, jobs, vehicles),
        )

    assert sample(101) == sample(101)
    assert {v for v, _ in sample(101)} == {"early", "tie"}
    assert [j for _, j in sample(101)] == list(jobs)
    assert len({sample(seed)[0][0] for seed in range(20)}) == 2


def test_nearest_rule_prefers_ready_then_prospective_without_masking_empty():
    def candidate(owner, distance, supply):
        fields = [0.0] * 12
        fields[1], fields[5] = distance, supply
        return Candidate(owner, DispatchTarget(owner, owner + "-port"), tuple(fields))

    candidates = (
        candidate("empty", 0, 0),
        candidate("processing", 1, 0.01),
        candidate("ready", 3, 0.01),
    )
    view = {
        "agvs": {"a": {"job": None}},
        "sources": {
            "empty": {"ready": []},
            "processing": {"ready": []},
            "ready": {"ready": ["j"]},
        },
    }
    request = DecisionRequest(0, "proposals", "dispatcher", "a", candidates, view)
    rule = RulePolicy("dispatcher", "nearest")
    assert rule.choose(request).action.owner == "ready"
    view["sources"]["ready"]["ready"] = []
    assert rule.choose(request).action.owner == "processing"
    request = replace(
        request,
        candidates=tuple(
            candidate(c.action.owner, c.features[1], 0) for c in candidates
        ),
    )
    assert rule.choose(request).action.owner == "empty"
    assert all(c.legal for c in request.candidates)


def test_old_action_contract_is_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        BoundaryCommand(contract="smartsom.production-actions/v3")


def test_automatic_progress_and_latches_replay_and_continue_exactly():
    import pickle

    original = replace(scenario(True), tick_limit=80)
    sim = ProductionSimulator(original, contract="v3")
    replay = ProductionSimulator(original, contract="v3")
    controller = driver(sim)
    for _ in range(15):
        row = controller.tick()
        replay_boundary(replay, row)
        assert sim.snapshot() == replay.snapshot()
    continued = pickle.loads(pickle.dumps(sim))  # in memory; no checkpoint file
    other = driver(continued)
    other.load_state_dict(controller.state_dict())
    for _ in range(20):
        row = controller.tick()
        assert row == other.tick()
        replay_boundary(replay, row)
        assert sim.snapshot() == replay.snapshot()


def test_target_change_loses_queue_place_same_target_preserves_it():
    sim, controller, owner, port = fixed_empty_source()
    controller.tick()
    for _ in range(45):
        controller.tick()
    vehicle = next(iter(sim.agvs))
    state = sim.agvs[vehicle]
    assert state["travel"] is None
    old_arrival = state["arrived_at"]
    state["empty_notified"] = False
    requests = sim.protocol.begin()
    targets = {
        r.owner: DispatchTarget(owner, port.port_id)
        for r in requests
        if r.role == "dispatcher"
    }
    sim.protocol.accept_proposals({}, targets)
    assert state["arrived_at"] == old_arrival
    sim.protocol.abort()
    state = sim.agvs[vehicle]
    state["empty_notified"] = False
    requests = sim.protocol.begin()
    targets = {
        r.owner: r.candidates[0].action for r in requests if r.role == "dispatcher"
    }
    assert targets[vehicle] != DispatchTarget(owner, port.port_id)
    sim.protocol.accept_proposals({}, targets)
    assert sim.agvs[vehicle]["arrived_at"] == sim.tick
    sim.protocol.abort()


def test_manual_midtrip_and_auto_nonphysical_override_fail_explicitly():
    original = scenario(True)
    matrix = original.transport_matrix
    changed = replace(
        matrix, times=tuple((a, b, 0 if a == b else 4) for a, b, _ in matrix.times)
    )
    with pytest.raises(ValueError, match="match geometry"):
        ProductionSimulator(replace(original, transport_matrix=changed), contract="v3")
    manual = replace(changed, source="manual")
    sim = ProductionSimulator(replace(original, transport_matrix=manual), contract="v3")
    vehicle = next(iter(sim.agvs))
    ports = list(sim.protocol.ports.values())
    sim.protocol.depart(vehicle, ports[0])
    before = deepcopy(sim.agvs[vehicle])
    with pytest.raises(ValueError, match="manual matrix cannot retarget"):
        sim.protocol.depart(vehicle, ports[1])
    assert sim.agvs[vehicle] == before


def test_manual_empty_target_same_selection_continues_to_arrival():
    original = scenario(True)
    matrix = replace(
        original.transport_matrix,
        source="manual",
        times=tuple(
            (a, b, 0 if a == b else 4) for a, b, _ in original.transport_matrix.times
        ),
    )
    sim = ProductionSimulator(replace(original, transport_matrix=matrix), contract="v3")
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    controller = driver(sim)
    controller.policies["dispatcher"].choose = lambda request: PolicyChoice(
        DispatchTarget(owner, port.port_id)
    )
    first = controller.tick()
    assert len(first["actions"]["dispatchers"]) == len(sim.agvs)
    trips = {v: deepcopy(a["travel"]) for v, a in sim.agvs.items()}
    assert all(t["arrival_tick"] == 4 and "path" not in t for t in trips.values())
    second = controller.tick()
    assert len(second["actions"]["dispatchers"]) == len(sim.agvs)
    assert all(a["travel"] == trips[v] for v, a in sim.agvs.items())
    for _ in range(5):
        arrived = sim.tick >= 4
        row = controller.tick()
        assert len(row["actions"]["dispatchers"]) == (len(sim.agvs) if arrived else 0)
    assert all(
        a["travel"] is None and a["point"] == port.port_id for a in sim.agvs.values()
    )


def test_manual_actual_midtrip_target_change_rolls_back_boundary():
    original = scenario(True)
    matrix = replace(
        original.transport_matrix,
        source="manual",
        times=tuple(
            (a, b, 0 if a == b else 4) for a, b, _ in original.transport_matrix.times
        ),
    )
    sim = ProductionSimulator(replace(original, transport_matrix=matrix), contract="v3")
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    controller = driver(sim)
    controller.policies["dispatcher"].choose = lambda request: PolicyChoice(
        DispatchTarget(owner, port.port_id)
    )
    controller.tick()
    before = sim.snapshot()
    controller.policies["dispatcher"].choose = lambda request: PolicyChoice(
        next(
            c.action
            for c in request.candidates
            if c.action != DispatchTarget(owner, port.port_id)
        )
    )
    with pytest.raises(ValueError, match="manual matrix cannot retarget"):
        controller.tick()
    assert sim.snapshot() == before and sim.protocol.stage == "closed"
