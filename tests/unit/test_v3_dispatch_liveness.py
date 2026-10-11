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
    sim = ProductionSimulator(scenario(jobs=6), contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    owner = sim.protocol.destination_owners(job)[0]
    port = loaded(sim, vehicle, job, owner)
    for j in list(sim.jobs)[1:5]:
        sim._remove(j)
        sim._place(j, owner, next(iter(sim.storage[owner])))
    processing = list(sim.jobs)[5]
    machine = next(m for m, pre in sim.pre.items() if pre == owner)
    sim._remove(processing)
    sim.jobs[processing].update(location=machine, slot=None)
    sim.machine_state[machine].update(
        job=processing, status="PROCESSING", remaining=100
    )
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
    assert physical["dispatch_semantics"] == "nonexclusive-intentions/3"
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
    session.parameters = {"train_every_ticks": 16, "target_update_ticks": 5}
    session.target_clock = {"g": start - start % 5}
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
    copies = []

    def optimize(groups):
        observed.append(session.ticks)
        if session.ticks - session.target_clock["g"] >= 5:
            copies.append(session.ticks)
            session.target_clock["g"] = session.ticks

    session.optimize_dqn = optimize
    while session.ticks < 48:
        session.advance_wave(48 - session.ticks)
    assert {t for t in (16, 32, 48) if t > start}.issubset(observed)
    assert session.cursor == 48
    assert copies == [t for t in range(5, 49, 5) if t > start]


def test_failed_cargo_is_not_admitted_to_output():
    sim = ProductionSimulator(scenario(), contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    sim.jobs[job].update(quality="FAIL", step=1)
    output = next(o for o, role in sim.roles.items() if role == "system_output")
    loaded(sim, vehicle, job, output)
    sim.protocol.begin()
    assert vehicle not in sim.protocol.priority_drops
    sim.protocol.abort()


def test_empty_transit_retains_one_original_opportunity():
    sim = ProductionSimulator(scenario("auto"), contract="v3")
    vehicle = next(iter(sim.agvs))
    owner = next(iter(sim.post.values()))
    port = sim.protocol.ports_for(owner, "pickup")[0]
    state = sim.agvs[vehicle]
    state["target"] = {"owner": owner, "port": port.port_id}
    sim.protocol.depart(vehicle, port)
    assert state["travel"] is not None
    requests = sim.protocol.begin()
    assert any(r.role == "dispatcher" and r.owner == vehicle for r in requests)
    assert state["empty_notified"]
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


def test_start_frees_capacity_before_arrived_unload_without_redirect():
    sim = ProductionSimulator(scenario(jobs=2), contract="v3")
    vehicle = next(iter(sim.agvs))
    waiting, cargo = list(sim.jobs)
    owner = sim.protocol.destination_owners(cargo)[0]
    machine = next(m for m, pre in sim.pre.items() if pre == owner)
    slot = next(iter(sim.storage[owner]))
    sim.capacity[owner][slot] = 1
    sim._remove(waiting)
    sim._place(waiting, owner, slot)
    loaded(sim, vehicle, cargo, owner)
    assert sim.machine_choices(machine)
    row = driver(sim).tick()
    assert not any(
        r["role"] == "dispatcher" and r["owner"] == vehicle for r in row["decisions"]
    )
    assert any(
        e["kind"] == "processing_started" and e["job"] == waiting for e in row["events"]
    )
    assert sim.jobs[cargo]["location"] == owner
    assert len(sim.storage[owner][slot]) == 1


def test_cross_port_last_slot_goes_to_earliest_arrival():
    sim = ProductionSimulator(scenario(jobs=2), contract="v3")
    first, second = list(sim.agvs)[:2]
    j1, j2 = list(sim.jobs)
    owner = sim.protocol.destination_owners(j1)[0]
    ports = sim.protocol.ports_for(owner, "drop_off")
    slot = next(iter(sim.storage[owner]))
    sim.capacity[owner][slot] = 1
    loaded(sim, first, j1, owner, arrival=10)
    loaded(sim, second, j2, owner, arrival=0)
    sim.agvs[second].update(
        target={"owner": owner, "port": ports[1].port_id},
        point=ports[1].port_id,
        cell=[ports[1].cell.x, ports[1].cell.y],
    )
    sim.protocol.begin()
    assert sim.protocol.priority_drops == {second}
    sim.protocol.abort()


def test_zero_time_new_arrival_cannot_replace_protected_drop():
    sim = ProductionSimulator(scenario(jobs=2), contract="v3")
    first, second = list(sim.agvs)[:2]
    j1, j2 = list(sim.jobs)
    owner = sim.protocol.destination_owners(j1)[0]
    loaded(sim, first, j1, owner, arrival=-1)
    loaded(sim, second, j2, owner, arrival=-2)
    other = next(
        p for p in sim.protocol.ports.values() if p.port_id != sim.agvs[first]["point"]
    )
    sim.agvs[second].update(
        target=None, point=other.port_id, cell=[other.cell.x, other.cell.y]
    )
    controller = driver(sim)
    choose = controller.policies["dispatcher"].choose
    port = sim.agvs[first]["target"]["port"]
    controller.policies["dispatcher"].choose = lambda request: (
        next(c.action for c in request.candidates if c.action.port == port)
        if request.owner == second
        else choose(request)
    )
    row = controller.tick()
    assert any(e["kind"] == "drop_started" and e["agv"] == first for e in row["events"])
    assert not any(
        e["kind"] == "drop_started" and e["agv"] == second for e in row["events"]
    )


@pytest.mark.parametrize("physical", [None, "old"])
def test_direct_restore_rejects_old_dispatch_before_mutation(physical):
    from types import SimpleNamespace

    from smartsom.experiments.composable import TrainingSession

    session = object.__new__(TrainingSession)
    session.prepared = SimpleNamespace(scenario=scenario())
    session.ticks, session.cursor = 7, 11
    state = {}
    if physical == "old":
        state["physical_contract"] = {
            **physical_contract(session.prepared.scenario),
            "dispatch_semantics": "nonexclusive-intentions/1",
        }
    before = dict(session.__dict__)
    with pytest.raises(ValueError, match="physical/dispatch contract"):
        session.restore(state)
    assert session.__dict__ == before


def test_selected_start_frees_correct_port_bound_slot_among_choices():
    from dataclasses import replace

    sim = ProductionSimulator(scenario(jobs=3), contract="v3")
    vehicle = next(iter(sim.agvs))
    first, selected, cargo = list(sim.jobs)
    owner = sim.protocol.destination_owners(cargo)[0]
    machine = next(m for m, pre in sim.pre.items() if pre == owner)
    original_slot = next(iter(sim.storage[owner]))
    sim.storage[owner]["extra"] = []
    sim.capacity[owner].update({original_slot: 1, "extra": 1})
    for job, slot in ((first, original_slot), (selected, "extra")):
        sim._remove(job)
        sim._place(job, owner, slot)
    port = loaded(sim, vehicle, cargo, owner)
    sim.protocol.ports[port.port_id] = replace(
        port,
        bindings=tuple(
            replace(binding, target=replace(binding.target, slot_id="extra"))
            if sim._target(binding.target)[0] == owner
            else binding
            for binding in port.bindings
        ),
    )
    controller = driver(sim)
    choose = controller.policies["machine"].choose
    controller.policies["machine"].choose = lambda request: (
        next(c.action for c in request.candidates if c.action[0] == selected)
        if request.owner == machine
        else choose(request)
    )
    row = controller.tick()
    assert not any(
        r["role"] == "dispatcher" and r["owner"] == vehicle for r in row["decisions"]
    )
    assert sim.storage[owner][original_slot] == [first]
    assert sim.storage[owner]["extra"] == [cargo]
    assert any(
        e["kind"] == "processing_started" and e["job"] == selected
        for e in row["events"]
    )


def test_repeated_machine_resolution_begin_abort_and_stage_only_restore():
    sim = ProductionSimulator(scenario(jobs=1), contract="v3")
    job = next(iter(sim.jobs))
    machine = next(iter(sim.machines))
    sim._remove(job)
    sim._place(job, sim.pre[machine], sim._free(sim.pre[machine]))
    before = deepcopy(sim.snapshot())
    protocol = sim.protocol
    requests = protocol.begin()
    commands = {
        r.owner: r.candidates[0].action for r in requests if r.role == "machine"
    }
    assert commands
    protocol.resolve_machines(commands)
    events = deepcopy(sim.events)
    resolved = deepcopy(sim.snapshot())
    protocol.resolve_machines(commands)
    assert sim.events == events
    assert sim.snapshot() == resolved
    with pytest.raises(ValueError, match="boundary is open"):
        protocol.begin()
    protocol.abort()
    assert sim.snapshot() == before
    controller = driver(sim)
    controller.tick(stage_only=True)
    assert sim.snapshot() == before
    row = controller.tick()
    assert sum(e["kind"] == "processing_started" for e in row["events"]) == 1
