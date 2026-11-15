"""HPOF: POST occupancy first, then resident PRE and machine job counts."""

from smartsom.representations.v2.facts import view


def choose_hpof(facts, request, candidates):
    if (request.group, request.stage) != ("dispatcher", "pickup"):
        raise ValueError("V3 HPOF is a Dispatcher pickup rule")
    v = view(facts)
    scored = []
    for candidate in candidates:
        owner = candidate.action[0]
        if facts.roles[owner] != "machine_post" or not facts.ready(owner):
            continue
        machine = facts.station[owner]
        backlog = len(facts.inventory(facts.pre[machine])) + int(
            facts.machine_state[machine]["job"] is not None
        )
        scored.append(((-v.occupancy(owner), -backlog, candidate.identity), candidate))
    return min(scored, key=lambda row: row[0])[1].identity if scored else None


def create_hpof():
    return choose_hpof


create_hpof.CONTRACT = "smartsom.rule/v1"
