"""Named global public context and semantic task-associated inputs."""

from smartsom.controllers.morris.observation import global_inputs


class Observation:
    def __init__(self, time_scale, count_scale):
        self.time_scale, self.count_scale = time_scale, count_scale

    def __call__(self, facts, request):
        state, owners, signature = global_inputs(
            facts, self.time_scale, self.count_scale
        )
        jobs = {r["id"]: r for r in facts["jobs"]}
        machines = {r["id"]: r for r in facts["machines"]}
        vehicles = {r["id"]: r for r in facts["vehicles"]}
        buffers = {r["id"]: r for r in facts["buffers"]}
        candidates = []
        associations = []
        for c in request.candidates:
            job, machine, vehicle, buffer = (
                jobs.get(c.job),
                machines.get(c.machine),
                vehicles.get(c.vehicle),
                buffers.get(c.destination),
            )
            associated_machine = c.machine or next(
                (
                    m["id"]
                    for m in machines.values()
                    if c.destination in (m["pre"], m["post"])
                ),
                None,
            )
            machine = machines.get(associated_machine)
            candidates.append(
                (
                    float(c.kind == "advance"),
                    float(c.kind == "process"),
                    float(c.kind == "transport"),
                    (job["due_at"] - facts["tick"]) / self.time_scale if job else 0.0,
                    float(job["rush"]) if job else 0.0,
                    len(job["remaining_operations"]) / self.count_scale if job else 0.0,
                    float(machine["status"] == "PROCESSING") if machine else 0.0,
                    machine["nominal"] / self.time_scale if machine else 0.0,
                    vehicle["cell"][0] / facts["topology"]["width"] if vehicle else 0.0,
                    vehicle["cell"][1] / facts["topology"]["height"]
                    if vehicle
                    else 0.0,
                    buffer["quota"] / self.count_scale if buffer else 0.0,
                    len(buffer["commitments"]) / self.count_scale if buffer else 0.0,
                    *(float(c.mode == mode) for mode in ("slow", "normal", "fast")),
                )
            )
            associations.append((associated_machine, c.vehicle, c.destination))
        return {
            "global": state,
            "candidates": tuple(candidates),
            "layout": signature,
            "owners": owners,
            "associations": tuple(associations),
        }


def create_observation(*, time_scale=200.0, count_scale=20.0):
    if time_scale <= 0 or count_scale <= 0:
        raise ValueError("observation scales must be positive")
    return Observation(time_scale, count_scale)


create_observation.OUTPUT_SCHEMA = "smartsom.e1a-current/actor-v2"


def create_critic_observation(*, time_scale=200.0, count_scale=20.0):
    if time_scale <= 0 or count_scale <= 0:
        raise ValueError("observation scales must be positive")

    def observe(facts, context):
        state, _, layout = global_inputs(facts, time_scale, count_scale)
        return {"global": state, "layout": layout}

    return observe


create_critic_observation.OUTPUT_SCHEMA = "smartsom.e1a-current/critic-v1"
