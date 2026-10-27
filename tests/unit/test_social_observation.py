"""Public, candidate-linked observations against constructed physical worlds."""

from dataclasses import asdict, replace

import pytest
from test_base_reservations import decide, options, place, sim_with

from smartsom.domain.base_decisions import BaseCandidate, vector
from smartsom.engine.social_observation import job_link, resource_row, vehicle_link


def configured(arm="all", **kwargs):
    sim = sim_with(**kwargs)
    sim.protocol.configure(options().model_copy(update={"social_arm": arm}))
    return sim


def test_machine_first_external_step_includes_down_full_stations():
    sim = configured(steps=("A", "A", "B", "C"))
    p = sim.protocol
    job = next(iter(sim.jobs))
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    place(sim, job, sim.pre[machine])
    targets = [
        m for m, spec in sorted(sim.machines.items()) if "B" in spec.operation_types
    ]
    sim.machine_state[targets[0]]["down"] = True
    candidates = [
        BaseCandidate(f"{job}:{mode}", (job, mode), vector(i))
        for i, mode in enumerate(("normal", "fast"))
    ]
    request = p.make_request("machine", machine, "process", candidates)
    first, second = [c.links[0] for c in request.social]
    assert first == second
    assert first.count == len(targets) and not first.terminal
    assert first.peers == tuple(resource_row(p, m) for m in targets)
    assert first.peers[0][0] == 1
    # No dependence on remote due-date/urgency/risk or hidden remaining work.
    before = asdict(request)
    for state in sim.machine_state.values():
        state["remaining"] = 76543
    for key, demand in list(sim.demands.items()):
        if key != sim.jobs[job]["demand"]:
            sim.demands[key] = replace(demand, due_at=999999, rush=True)
    assert asdict(p.make_request("machine", machine, "process", candidates)) == before


def test_terminal_and_pre_alternatives_have_distinct_relations():
    sim = configured(steps=("A", "A"))
    p = sim.protocol
    job = next(iter(sim.jobs))
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    car = next(iter(sim.agvs))
    place(sim, job, car)
    request = p.make_request(
        "pre",
        sim.pre[machine],
        "accept",
        (
            BaseCandidate("accept", (car,), vector()),
            BaseCandidate("STOP", ("STOP",), vector()),
        ),
        applicants=(car,),
    )
    downstream, alternative = request.social[0].links
    assert downstream.terminal and downstream.peers == ()
    assert not alternative.terminal and alternative.count == 1
    assert request.social[1].links == request.social[0].links
    assert alternative.peers == tuple(
        resource_row(p, m)
        for m, spec in sorted(sim.machines.items())
        if m != machine and "A" in spec.operation_types
    )


@pytest.mark.parametrize("loaded", (False, True))
def test_peer_wait_starts_at_arrival_not_reservation_and_has_no_job_details(loaded):
    sim = configured(n=8)
    p = sim.protocol
    own, peer, unrelated = list(sim.agvs)[:3]
    if loaded:
        for v, j in zip((own, peer, unrelated), list(sim.jobs)[:3], strict=True):
            place(sim, j, v)
        target = p.candidate_owners(peer)[0]
    else:
        target = p.sources[0]
    p.accept(peer, target)
    sim.tick = 10
    link = vehicle_link(p, target, own, loaded)
    assert len(link.peers) == 1 and link.count == 0
    assert link.peers[0][:2] == (1.0, 0.0)
    assert link.peers[0][3] == 0
    sim.agvs[peer]["cell"] = list(p.routes.ports[target][0])
    sim.agvs[peer]["arrived_at"] = 7
    row = vehicle_link(p, target, own, loaded).peers[0]
    assert row[:4] == (0.0, 1.0, 0.0, 3 / p.time_scale)
    assert row[4:] == (0.0,) * 44
    # An uncommitted car and another car's carried job urgency are not features.
    sim.agvs[unrelated]["cell"] = [1, 1]
    if loaded:
        j = sim.agvs[peer]["job"]
        d = sim.jobs[j]["demand"]
        sim.demands[d] = replace(sim.demands[d], due_at=9999, rush=True)
    assert vehicle_link(p, target, own, loaded).peers[0] == row
    sim.agvs[peer]["service"] = {"kind": "drop_off" if loaded else "pickup"}
    assert vehicle_link(p, target, own, loaded).peers == ()


def test_post_reservation_is_unbound_but_site_pair_is_job_linked():
    sim = configured(steps=("A", "B", "C"))
    p = sim.protocol
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    jobs = list(sim.jobs)[:2]
    place(sim, jobs[0], sim.pre[machine])
    place(sim, jobs[1], sim.post[machine])
    sim.jobs[jobs[1]]["step"] = 2
    owner = sim.post[machine]
    cars = list(sim.agvs)[:2]
    request = p.make_request(
        "post",
        owner,
        "accept",
        tuple(BaseCandidate(f"accept:{v}", (v,), vector()) for v in cars)
        + (BaseCandidate("STOP", ("STOP",), vector()),),
        applicants=cars,
    )
    assert all(c.links == request.social[0].links for c in request.social)
    assert {link.identity for link in request.social[0].links} == {
        f"downstream:{j}" for j in jobs
    }
    site = p.make_request(
        "post",
        owner,
        "service",
        tuple(BaseCandidate(f"pair:{j}", (j, cars[0]), vector()) for j in jobs),
    )
    assert all(len(c.links) == 1 for c in site.social)
    assert site.social[1].links == (job_link(p, jobs[1]),)
    assert site.social[0].links[0].peers != site.social[1].links[0].peers


def test_disabled_payload_and_projection_do_not_change_actions_or_world():
    base, sl = sim_with(), configured()
    a, b = decide(base), decide(sl)
    assert [(r.group, r.stage, r.candidates, r.sets, r.context) for r in a] == [
        (r.group, r.stage, r.candidates, r.sets, r.context) for r in b
    ]
    assert all(r.social is None for r in a)
    assert all(r.social is not None for r in b)
    assert base.jobs == sl.jobs and base.agvs == sl.agvs
    assert base.machine_state == sl.machine_state


def test_projection_preserves_full_rule_trajectory():
    import io
    import json

    from test_base_reservations import workload

    from smartsom.algorithms.base_rules import BaseRule
    from smartsom.config.base_layout import base_factory
    from smartsom.experiments.base_diagnostic import episode

    outcomes = []
    traces = []
    for arm in ("base", "all"):
        trace = io.StringIO()
        result, rewards, critic = episode(
            base_factory("small"),
            workload(),
            options().model_copy(update={"social_arm": arm}),
            BaseRule(),
            19,
            "high",
            100,
            max_ticks=150,
            trace=trace,
        )
        result.pop("wall_seconds")
        outcomes.append((result, rewards, critic))
        records = [json.loads(line) for line in trace.getvalue().splitlines()]
        for record in records:
            for choice in record["decisions"]:
                choice["decision"].pop("social")
        traces.append(records)
    assert outcomes[0] == outcomes[1]
    assert traces[0] == traces[1]


def test_arrival_wait_resets_on_real_leave_and_reentry():
    from smartsom.domain.production import MOVES

    sim = configured(n=8)
    p = sim.protocol
    own, peer = list(sim.agvs)[:2]
    source = p.sources[0]
    occupied = {tuple(a["cell"]) for a in sim.agvs.values()}
    port, action, neighbor = next(
        (port, action, (port[0] + dx, port[1] + dy))
        for port in p.routes.ports[source]
        for action, (dx, dy) in MOVES.items()
        if (port[0] + dx, port[1] + dy) in p.routes.neighbors
        and (port[0] + dx, port[1] + dy) not in occupied
        and (port[0] + dx, port[1] + dy) not in p.routes.ports[source]
        and port not in occupied
    )
    sim.agvs[peer]["cell"] = list(port)
    sim.tick = 5
    p.accept(peer, source)
    assert vehicle_link(p, source, own, False).peers[0][3] == 0
    sim.tick = 9
    assert vehicle_link(p, source, own, False).peers[0][3] == 4 / p.time_scale
    moves = dict.fromkeys(sim.agvs, "WAIT")
    moves[peer] = action
    assert not p.advance(moves)["rejections"]
    assert tuple(sim.agvs[peer]["cell"]) == neighbor
    assert vehicle_link(p, source, own, False).peers[0][3] == 0
    dx, dy = MOVES[action]
    moves[peer] = next(a for a, delta in MOVES.items() if delta == (-dx, -dy))
    assert not p.advance(moves)["rejections"]
    assert sim.agvs[peer]["arrived_at"] == sim.tick
    assert vehicle_link(p, source, own, False).peers[0][3] == 0
