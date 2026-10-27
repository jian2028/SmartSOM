"""Project approved peer fields at the same boundary as a Base request.

This reads public facts from the protocol, never predicts outcomes or mutates
the world. Tuples detach all actor data before the policy receives it.
"""

from smartsom.domain.base_decisions import vector
from smartsom.domain.social import SocialCandidate, SocialLink


def resource_row(protocol, machine):
    core = protocol.core
    state = core.machine_state[machine]
    pre, post = core.pre[machine], core.post[machine]
    count, time = protocol.count_scale, protocol.time_scale
    return vector(
        state["down"],
        state["status"] == "PROCESSING",
        state["status"] == "BLOCKED",
        len(protocol.items(pre)) / count,
        len(protocol.items(post)) / count,
        len(protocol.ready(post)) / count,
        sum(core.capacity[pre].values()) / count,
        sum(core.capacity[post].values()) / count,
        len(protocol.commitments(pre)) / count,
        len(protocol.commitments(post)) / count,
        protocol.quota(pre) / count,
        protocol.quota(post) / count,
        sum(protocol.nominal(j, machine) for j in protocol.items(pre)) / time,
        protocol.current_standard_work(machine) / time,
    )


def job_link(protocol, job, machine=None, *, alternative=False, resource=False):
    core = protocol.core
    row = core.jobs[job]
    steps = core.demands[row["demand"]].steps
    index = row["step"]
    if not alternative and machine and row["location"] != core.post[machine]:
        while index < len(steps) and (
            steps[index].operation_type in core.machines[machine].operation_types
        ):
            index += 1
    terminal = index == len(steps)
    machines = (
        ()
        if terminal
        else tuple(
            m
            for m, spec in sorted(core.machines.items())
            if steps[index].operation_type in spec.operation_types
            and (not alternative or m != machine)
        )
    )
    peers = tuple(
        peer
        for m in machines
        for peer in (
            (resource_row(protocol, m),)
            if resource
            else protocol.station_vectors(core.pre[m])
        )
    )
    return SocialLink(
        f"{'alternative' if alternative else 'downstream'}:{job}",
        "resource" if resource else "station",
        protocol.job_vector(job),
        peers,
        terminal,
        len(machines),
    )


def vehicle_link(protocol, owner, vehicle, loaded):
    core = protocol.core
    peers = []
    for other in protocol.commitments(owner):
        a = core.agvs[other]
        if other == vehicle or a["service"] or bool(a["job"]) != loaded:
            continue
        arrived = tuple(a["cell"]) in protocol.routes.ports[owner]
        peers.append(
            vector(
                not arrived,
                arrived,
                protocol.routes.to_owner(a["cell"], owner) / protocol.distance_scale,
                (core.tick - max(a["arrived_at"], a["reservation_tick"]))
                / protocol.time_scale
                if arrived
                else 0.0,
            )
        )
    return SocialLink(f"reservations:{owner}", "vehicle", vector(), tuple(peers))


def project_social(protocol, request):
    core = protocol.core
    group, owner, stage = request.group, request.owner, request.stage
    machine = owner if group == "machine" else protocol.station.get(owner)
    local = ()
    if group in ("post", "input") and stage == "accept":
        jobs = protocol.station_items(machine) if machine else protocol.items(owner)
        local = tuple(job_link(protocol, j, machine) for j in sorted(jobs))
    pre_links = {}
    if group == "pre":
        for candidate in request.candidates:
            if candidate.action != ("STOP",):
                job = core.agvs[candidate.action[0]]["job"]
                pre_links[candidate.identity] = (
                    job_link(protocol, job, machine),
                    job_link(protocol, job, machine, alternative=True, resource=True),
                )
    result = []
    for candidate in request.candidates:
        action = candidate.action
        if group == "dispatcher":
            links = (
                ()
                if action == ("EMPTY_WAIT",)
                else (vehicle_link(protocol, action[0], owner, stage == "delivery"),)
            )
        elif group == "machine":
            links = (job_link(protocol, action[0], machine, resource=True),)
        elif group == "pre":
            links = (
                tuple(link for rows in pre_links.values() for link in rows)
                if action == ("STOP",)
                else pre_links[candidate.identity]
            )
        elif stage == "accept":
            links = local
        else:
            links = (job_link(protocol, action[0]),)
        result.append(SocialCandidate(candidate.identity, links))
    return tuple(result)
