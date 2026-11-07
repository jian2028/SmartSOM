"""Pure current public-facts projection, with one shared Base implementation."""

import math

from .features import SCHEMA, inventory, related_jobs


def create_observation(*, social=False, count_scale=20.0, time_scale=200.0):
    if type(social) is not bool:
        raise ValueError("social must be a boolean")
    if not all(math.isfinite(x) and x > 0 for x in (count_scale, time_scale)):
        raise ValueError("observation scales must be positive and finite")

    def encode(facts, decision):
        jobs = {row["id"]: row for row in facts["jobs"]}
        vehicles = {row["id"]: row for row in facts["vehicles"]}
        buffers = {row["id"]: row for row in facts["buffers"]}
        machines = {row["id"]: row for row in facts["machines"]}
        if decision.group == "machine":
            machine = machines[decision.owner]
            local = set(inventory(buffers[machine["pre"]]))
            local.update(inventory(buffers[machine["post"]]))
            if machine["job"]:
                local.add(machine["job"])
        elif decision.group == "dispatcher":
            job = vehicles[decision.owner]["job"]
            local = (job,) if job else ()
        else:
            local = inventory(buffers[decision.owner])
        base = (
            len(local) / count_scale,
            sum(jobs[j]["due_at"] < facts["tick"] for j in local) / count_scale,
        )
        candidates, peers = [], []
        for candidate in decision.candidates:
            related = related_jobs(decision, candidate.action, jobs, vehicles, buffers)
            candidates.append(
                (
                    len(related) / count_scale,
                    sum(jobs[j]["due_at"] - facts["tick"] for j in related)
                    / max(1, len(related))
                    / time_scale,
                )
            )
            if social:
                operations = {
                    jobs[j]["remaining_operations"][0]["type"]
                    for j in related
                    if jobs[j]["remaining_operations"]
                }
                linked = [
                    m
                    for m in machines.values()
                    if m["id"] != decision.owner
                    and decision.owner not in (m["pre"], m["post"])
                    and operations.intersection(m["capabilities"])
                ]
                peers.append(
                    (
                        sum(m["status"] == "PROCESSING" for m in linked)
                        / max(1, len(linked)),
                        sum(len(inventory(buffers[m["pre"]])) for m in linked)
                        / count_scale,
                    )
                )
            else:
                peers.append(())
        return {"base": base, "candidates": tuple(candidates), "social": tuple(peers)}

    return encode


create_observation.OUTPUT_SCHEMA = SCHEMA
