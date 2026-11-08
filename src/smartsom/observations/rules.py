"""Named rule inputs, independent of any neural feature-vector layout."""

import math
from functools import cache


def candidate_facts(facts, request):
    time, distance, count = facts.scales
    result = {}

    @cache
    def job_values(job):
        row = facts.jobs[job]
        demand = facts.demands[row["demand"]]
        locked = job in facts.service_jobs
        queued = row["location"] in facts.storage and not locked
        return {
            "due_slack": (demand.due_at - facts.tick) / time,
            "remaining_work": facts.remaining_work(job) / time,
            "queue_age": (facts.tick - row["since"]) / time if queued else 0.0,
        }

    for candidate in request.candidates:
        action = candidate.action
        if action in (("STOP",), ("EMPTY_WAIT",)):
            result[candidate.identity] = {}
            continue
        row = {}
        if request.group == "machine":
            job, mode = action
            row.update(
                job_values(job),
                nominal_duration=facts.nominal(job, request.owner, mode) / time,
            )
        elif request.group == "dispatcher":
            (owner,) = action
            machine = facts.station.get(owner)
            jobs = (
                facts.ready(owner)
                if request.stage == "pickup"
                else facts.inventory(facts.pre[machine])
                if machine
                else ()
            )
            row.update(
                distance=facts.routes.to_owner(facts.agvs[request.owner]["cell"], owner)
                / distance,
                ready_count=len(jobs) / count,
                ready_min_due=min(
                    (
                        (facts.demands[facts.jobs[j]["demand"]].due_at - facts.tick)
                        / time
                        for j in jobs
                    ),
                    default=0.0,
                ),
                commitments=len(facts.commitments(owner)) / count,
                machine_destination=machine is not None,
                pre_count=len(facts.inventory(facts.pre[machine])) / count
                if machine
                else 0.0,
                pre_work=sum(
                    facts.nominal(j, machine)
                    for j in facts.inventory(facts.pre[machine])
                )
                / time
                if machine
                else 0.0,
            )
        else:
            vehicle = action[-1]
            state = facts.agvs[vehicle]
            row.update(
                distance=facts.routes.to_owner(state["cell"], request.owner) / distance,
                arrival_age=(facts.tick - state["arrived_at"]) / time,
                idle_age=(facts.tick - facts.idle_since.get(vehicle, facts.tick))
                / time,
            )
            job = (
                action[0]
                if request.stage == "service" and request.group in ("post", "input")
                else state["job"]
            )
            if job:
                row.update(job_values(job))
            if request.stage == "service" and request.group in ("post", "input"):
                j = facts.jobs[job]
                steps = facts.demands[j["demand"]].steps
                terminal = j["step"] >= len(steps)
                feasible = []
                if not terminal:
                    operation = steps[j["step"]].operation_type
                    for machine, spec in facts.machines.items():
                        pre = facts.pre[machine]
                        if (
                            operation in spec.operation_types
                            and not facts.machine_state[machine]["down"]
                            and facts.quota(pre) > 0
                            and math.isfinite(facts.routes.to_owner(state["cell"], pre))
                        ):
                            feasible.append(
                                sum(
                                    facts.nominal(k, machine)
                                    for k in facts.inventory(pre)
                                )
                            )
                row.update(
                    feasible_next_count=len(feasible) / count,
                    next_pre_work=min(feasible, default=0) / time,
                    has_feasible=bool(feasible),
                    terminal=terminal,
                )
        result[candidate.identity] = row
    return result


def create_observation():
    def observe(facts, request):
        return {"candidates": candidate_facts(facts, request)}

    return observe


create_observation.OUTPUT_SCHEMA = "smartsom.named-rule-inputs/v1"


def create_random_observation():
    return lambda facts, request: {}


create_random_observation.OUTPUT_SCHEMA = "smartsom.no-features/v1"
