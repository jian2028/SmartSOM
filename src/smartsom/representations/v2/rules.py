"""V2 rule catalog; stateless semantic selectors over current legal candidates."""

from smartsom.representations.v2.facts import view

CATALOG = {
    "machine.process": ("EDD", "SPT", "MDD", "SLACK", "SRPT", "CR", "FIFO"),
    "pre.accept": ("EDD", "SPT", "MDD", "SLACK", "SRPT", "CR", "NVF"),
    "pre.service": ("EDD", "SPT", "MDD", "SLACK", "SRPT", "CR", "FCFS"),
    "post.accept": ("NVF", "LIV", "LUV", "LCL_SD", "HCL_SD"),
    "input.accept": ("NVF", "LIV", "LUV", "LCL_SD", "HCL_SD"),
    "post.service": ("EDD", "MDD", "SLACK", "SRPT", "LWT", "LNW"),
    "input.service": ("EDD", "MDD", "SLACK", "SRPT", "LWT", "LNW"),
    "dispatcher.pickup": (
        "NPF",
        "LWF",
        "EDSF",
        "HPOF",
        "LPCL",
        "MRSF",
        "HPCL",
        "EVR_P",
    ),
    "dispatcher.delivery": (
        "NDF",
        "LPWF",
        "LPOF",
        "LDCL",
        "SPTD",
        "LPTD",
        "LRMF",
        "STPT",
        "EVR_D",
    ),
}


def score(v, request, candidate, rule):
    f = v.f
    group, stage, action = request.group, request.stage, candidate.action
    if group == "dispatcher":
        owner = action[0]
        distance = f.routes.to_owner(f.agvs[request.owner]["cell"], owner)
        if rule in ("NPF", "NDF"):
            return distance
        if stage == "pickup":
            ready = f.ready(owner)
            if rule == "LWF":
                return -max((v.job(j)[2] for j in ready), default=0)
            if rule == "EDSF":
                supply = v.supply(owner)
                return min(v.job(j)[0] for j in supply) if supply else None
            if rule == "HPOF":
                return (
                    -v.occupancy(owner)
                    if f.roles[owner] == "machine_post" and ready
                    else None
                )
            if rule == "MRSF":
                return -len(ready)
            return v.density(owner, request.owner) * (-1 if rule == "HPCL" else 1)
        machine = f.station.get(owner)
        if rule == "LDCL":
            return v.density(owner, request.owner)
        if machine is None:
            return None
        job = f.agvs[request.owner]["job"]
        duration = f.nominal(job, machine)
        return {
            "LPWF": v.pre_work(machine),
            "LPOF": v.occupancy(owner),
            "SPTD": duration,
            "LPTD": -duration,
            "LRMF": f.current_standard_work(machine),
            "STPT": distance + duration,
        }[rule]
    if group in ("post", "input") and stage == "accept":
        vehicle = action[0]
        state = f.agvs[vehicle]
        distance = f.routes.to_owner(state["cell"], request.owner)
        if rule == "NVF":
            return distance
        if rule == "LIV":
            return -(f.tick - f.idle_since.get(vehicle, f.tick))
        if rule == "LUV":
            return state["task_ticks"] / f.tick if f.tick else 0.0
        density = v.density(vehicle=vehicle)
        return (density * (-1 if rule == "HCL_SD" else 1), distance)
    job = action[0] if group != "pre" else f.agvs[action[0]]["job"]
    due, work, wait = v.job(job)
    if rule == "CR":
        return due / work if work > 0 else None
    if rule == "SPT":
        return f.nominal(
            job, request.owner if group == "machine" else f.station[request.owner]
        )
    if rule == "NVF":
        return f.routes.to_owner(f.agvs[action[0]]["cell"], request.owner)
    if rule == "FCFS":
        return f.agvs[action[0]]["arrived_at"]
    if rule == "LNW":
        _, feasible, terminal = v.destinations(job, action[1])
        if terminal or not feasible:
            return None
        return min(v.pre_work(f.station[owner]) for owner, _ in feasible)
    return {
        "EDD": due,
        "MDD": max(due, work),
        "SLACK": due - work,
        "SRPT": work,
        "FIFO": -wait,
        "LWT": -wait,
    }[rule]


def choose(facts, request, candidates, rule):
    v = view(facts)
    if request.group in ("post", "input") and request.stage == "service":
        facts.component_cache["smartsom.v2.service-candidates"] = candidates
        candidates = tuple(
            v.pair(job, candidates)[0]
            for job in sorted({c.action[0] for c in candidates})
        )
    if rule.startswith("EVR_"):
        members = (
            ("NPF", "LWF", "EDSF", "HPOF", "LPCL")
            if rule == "EVR_P"
            else ("NDF", "LPWF", "LPOF", "LDCL")
        )
        votes = {c.identity: 0.0 for c in candidates}
        for member in members:
            scored = [(c, score(v, request, c, member)) for c in candidates]
            scored = [(c, s) for c, s in scored if s is not None]
            if len({s for _, s in scored}) <= 1:
                continue
            best = min(s for _, s in scored)
            winners = [c for c, s in scored if s == best]
            for candidate in winners:
                votes[candidate.identity] += 1 / len(winners)
        return (
            min(
                candidates,
                key=lambda c: (
                    -votes[c.identity],
                    facts.routes.to_owner(
                        facts.agvs[request.owner]["cell"], c.action[0]
                    ),
                    c.identity,
                ),
            ).identity
            if candidates
            else None
        )
    scored = [(c, score(v, request, c, rule)) for c in candidates]
    scored = [(c, s) for c, s in scored if s is not None]
    return (
        min(scored, key=lambda row: (row[1], row[0].identity))[0].identity
        if scored
        else None
    )


def create_rule(*, rule):
    if rule not in {r for rules in CATALOG.values() for r in rules}:
        raise ValueError(f"unknown V2 rule: {rule}")
    return lambda facts, request, candidates: choose(facts, request, candidates, rule)


create_rule.CONTRACT = "smartsom.rule/v1"
