"""Arrived-service priority and persistent empty intent regression checks."""

from copy import deepcopy

import pytest
from test_travel_time_matrix import driver, scenario

from smartsom.domain.travel_time import physical_contract, validate_model_contract
from smartsom.engine.production import ProductionSimulator


def loaded(sim, vehicle, job, owner, *, arrival=0):
    sim._remove(job)
    sim.jobs[job].update(location=vehicle, slot=None)
    port = sim.protocol.ports_for(owner, "drop_off")[0]
    sim.agvs[vehicle].update(
        job=job,
        target={"owner": owner, "port": port.port_id},
        point=port.port_id,
        cell=[port.cell.x, port.cell.y],
        arrived_at=arrival,
        travel=None,
    )
    return port


@pytest.mark.parametrize("mode", ["auto", "zero"])
@pytest.mark.parametrize("quality", ["UNKNOWN", "PASS"])
def test_arrived_loaded_service_precedes_dispatch(mode, quality):
    sim = ProductionSimulator(scenario(mode), contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    sim.jobs[job].update(quality=quality, step=1)
    owner = (
        next(iter(sim.stations))
        if quality == "UNKNOWN"
        else sim.protocol.destination_owners(job)[0]
    )
    loaded(sim, vehicle, job, owner)
    row = driver(sim).tick()
    assert not any(
        r["role"] == "dispatcher" and r["owner"] == vehicle for r in row["decisions"]
    )
    assert any(
        e["kind"] == "drop_started" and e["agv"] == vehicle for e in row["events"]
    )
    assert sim.agvs[vehicle]["job"] is None
    assert sim.jobs[job]["location"] != vehicle


def test_incompatible_output_does_not_suppress_redirect():
    sim = ProductionSimulator(scenario(), contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    sim.jobs[job].update(quality="PASS", step=0)
    output = next(o for o, role in sim.roles.items() if role == "system_output")
    loaded(sim, vehicle, job, output)
    requests = sim.protocol.begin()
    assert any(r.role == "dispatcher" and r.owner == vehicle for r in requests)
    assert vehicle not in sim.protocol.priority_drops
    sim.protocol.abort()


def test_queue_head_has_priority_and_waiter_can_redirect():
    sim = ProductionSimulator(scenario(jobs=2), contract="v3")
    first, second = list(sim.agvs)[:2]
    j1, j2 = list(sim.jobs)
    owner = sim.protocol.destination_owners(j1)[0]
    loaded(sim, first, j1, owner, arrival=-1)
    loaded(sim, second, j2, owner)
    requests = sim.protocol.begin()
    assert first in sim.protocol.priority_drops
    assert second not in sim.protocol.priority_drops
    assert any(r.role == "dispatcher" and r.owner == second for r in requests)
    sim.protocol.abort()


def test_full_bound_slot_keeps_redirect_opportunity():
    sim = ProductionSimulator(scenario(jobs=5), contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    owner = sim.protocol.destination_owners(job)[0]
    port = loaded(sim, vehicle, job, owner)
    for j in list(sim.jobs)[1:]:
        sim._remove(j)
        sim._place(j, owner, next(iter(sim.storage[owner])))
    requests = sim.protocol.begin()
    assert (
        sim.protocol.drop_slot(owner, __import__("collections").Counter(), port) is None
    )
    assert any(r.role == "dispatcher" and r.owner == vehicle for r in requests)
    sim.protocol.abort()


def test_repeated_empty_target_preserves_arrival_and_gets_future_decisions():
    sim = ProductionSimulator(scenario("auto"), contract="v3")
    vehicle = next(iter(sim.agvs))
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    state = sim.agvs[vehicle]
    state.update(
        target={"owner": owner, "port": port.port_id},
        point=port.port_id,
        cell=[port.cell.x, port.cell.y],
        arrived_at=-7,
        empty_notified=True,
    )
    before = deepcopy(state)
    controller = driver(sim)
    choose = controller.policies["dispatcher"].choose
    controller.policies["dispatcher"].choose = lambda request: (
        next(c.action for c in request.candidates if c.action.port == port.port_id)
        if request.owner == vehicle
        else choose(request)
    )
    for _ in range(3):
        row = controller.tick()
        assert any(
            r["role"] == "dispatcher" and r["owner"] == vehicle
            for r in row["decisions"]
        )
        assert state["target"] == before["target"]
        assert state["cell"] == before["cell"]
        assert state["arrived_at"] == -7
        assert state["travel"] is None


def test_nonempty_owner_and_transit_keep_commitment():
    sim = ProductionSimulator(scenario("auto"), contract="v3")
    vehicle = next(iter(sim.agvs))
    job = next(iter(sim.jobs))
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    sim._remove(job)
    sim._place(job, owner, next(iter(sim.storage[owner])))
    sim.agvs[vehicle].update(
        target={"owner": owner, "port": port.port_id}, empty_notified=True
    )
    requests = sim.protocol.begin()
    assert not any(r.role == "dispatcher" and r.owner == vehicle for r in requests)
    sim.protocol.abort()


def test_old_dispatch_contract_requires_retraining():
    from smartsom.domain.production_decisions import (
        ACTION_CONTRACT,
        OBSERVATION_CONTRACT,
    )

    case = scenario()
    physical = physical_contract(case)
    assert physical["dispatch_semantics"] == "nonexclusive-intentions/2"
    metadata = dict(
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
        physical_contract={
            **physical,
            "dispatch_semantics": "nonexclusive-intentions/1",
        },
    )
    with pytest.raises(ValueError, match="retraining"):
        validate_model_contract(metadata, case)


@pytest.mark.parametrize("envs", [1, 3, 5])
@pytest.mark.parametrize("start", [0, 15, 17])
def test_parallel_dqn_waves_reach_each_optimizer_boundary(monkeypatch, envs, start):
    from types import SimpleNamespace

    import smartsom.experiments.composable as module
    from smartsom.experiments.composable import TrainingSession

    session = object.__new__(TrainingSession)
    session.parallel_sampling = True
    session.settings = SimpleNamespace(algorithm="dqn", groups=())
    session.parameters = {"train_every_ticks": 16}
    session.ticks = session.cursor = start
    session.policies, session.routes, session.matching = {}, {}, "first_arrival"
    session.sims = [
        SimpleNamespace(done=False, protocol=SimpleNamespace(public_view=lambda: {}))
        for _ in range(envs)
    ]
    session.episodes = [0] * envs
    session.sampler_states = [{} for _ in range(envs)]
    session.collector = SimpleNamespace(drain=lambda algorithm: {})
    session.executor = SimpleNamespace(
        submit=lambda fn, sim, partners, *args: SimpleNamespace(
            result=lambda: (sim, {}, [], partners)
        )
    )
    monkeypatch.setattr(
        module,
        "BoundaryCoordinator",
        lambda *args, **kwargs: SimpleNamespace(load_state_dict=lambda state: None),
    )

    def finish(*args, **kwargs):
        session.ticks += 1

    session._finish_tick = finish
    observed = []
    session.optimize_dqn = lambda groups: observed.append(session.ticks)
    while session.ticks < 48:
        session.advance_wave(48 - session.ticks)
    assert {t for t in (16, 32, 48) if t > start}.issubset(observed)
    assert session.cursor == 48


def test_failed_cargo_is_not_admitted_to_output():
    sim = ProductionSimulator(scenario(), contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    sim.jobs[job].update(quality="FAIL", step=1)
    output = next(o for o, role in sim.roles.items() if role == "system_output")
    loaded(sim, vehicle, job, output)
    sim.protocol.begin()
    assert vehicle not in sim.protocol.priority_drops
    sim.protocol.abort()


def test_empty_transit_never_retargets():
    sim = ProductionSimulator(scenario("auto"), contract="v3")
    vehicle = next(iter(sim.agvs))
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    state = sim.agvs[vehicle]
    state["target"] = {"owner": owner, "port": port.port_id}
    sim.protocol.depart(vehicle, port)
    assert state["travel"] is not None
    requests = sim.protocol.begin()
    assert not any(r.role == "dispatcher" and r.owner == vehicle for r in requests)
    sim.protocol.abort()


def test_same_owner_different_port_is_real_reroute():
    sim = ProductionSimulator(scenario("auto"), contract="v3")
    vehicle = next(iter(sim.agvs))
    owner = next(iter(sim.post.values()))
    first, second = sim.protocol.ports_for(owner, "pickup")
    state = sim.agvs[vehicle]
    state.update(
        target={"owner": owner, "port": first.port_id},
        point=first.port_id,
        cell=[first.cell.x, first.cell.y],
        arrived_at=-7,
        empty_notified=True,
    )
    controller = driver(sim)
    choose = controller.policies["dispatcher"].choose
    controller.policies["dispatcher"].choose = lambda request: (
        next(c.action for c in request.candidates if c.action.port == second.port_id)
        if request.owner == vehicle
        else choose(request)
    )
    controller.tick()
    assert state["target"] == {"owner": owner, "port": second.port_id}
    assert state["arrived_at"] == 0
    assert state["travel"] is not None
    assert state["travel"]["from"] == first.port_id
    assert state["travel"]["to"] == second.port_id


def test_busy_port_keeps_loaded_redirect_opportunity():
    sim = ProductionSimulator(scenario(jobs=2), contract="v3")
    first, second = list(sim.agvs)[:2]
    j1, j2 = list(sim.jobs)
    owner = sim.protocol.destination_owners(j1)[0]
    loaded(sim, first, j1, owner)
    loaded(sim, second, j2, owner)
    sim.agvs[first]["service"] = {
        "kind": "drop",
        "job": j1,
        "owner": owner,
        "slot": next(iter(sim.storage[owner])),
        "started_at": 0,
        "remaining": 1,
    }
    requests = sim.protocol.begin()
    assert second not in sim.protocol.priority_drops
    assert any(r.role == "dispatcher" and r.owner == second for r in requests)
    sim.protocol.abort()


def test_dqn_rejects_replay_capacity_that_can_never_supply_batch():
    from smartsom.config.experiment_v3 import DQNParameters

    with pytest.raises(ValueError, match="replay_capacity must be at least batch_size"):
        DQNParameters(replay_capacity=63, batch_size=64)
    assert DQNParameters(replay_capacity=64, batch_size=64).batch_size == 64
