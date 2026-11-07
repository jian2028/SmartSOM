"""Fixed-layout public state; no slots for future jobs or realization tables."""

import hashlib
import json

from smartsom.algorithms.base_policy import DECISION_POINTS
from smartsom.domain.public_facts import plain
from smartsom.observations.base import Projection

POINTS = tuple(sorted(DECISION_POINTS))


def one_hot(value, keys):
    return tuple(float(value == key) for key in keys)


class Observation:
    def __init__(self, *, time_scale=200.0, count_scale=20.0):
        if time_scale <= 0 or count_scale <= 0:
            raise ValueError("observation scales must be positive")
        self.time_scale, self.count_scale = time_scale, count_scale

    def __call__(self, facts, request):
        local = Projection(facts, request).build()
        state, owners, signature = global_inputs(
            facts, self.time_scale, self.count_scale
        )
        token = (
            *one_hot(request.group + "." + request.stage, POINTS),
            *one_hot(request.owner, owners),
        )
        return {
            "context": (*state, *local.context, *token),
            "candidates": local.candidates,
            "token": token,
            "layout": signature,
            "prefix": (),
            "prefix_actions": (),
        }


def create_observation(**params):
    return Observation(**params)


create_observation.OUTPUT_SCHEMA = "smartsom.central-rule-observation/v1"


def global_inputs(facts, t, n):
    machines = sorted(facts["machines"], key=lambda r: r["id"])
    vehicles = sorted(facts["vehicles"], key=lambda r: r["id"])
    buffers = sorted(facts["buffers"], key=lambda r: r["id"])
    owners = tuple(sorted(r["id"] for r in (*machines, *vehicles, *buffers)))
    topology = facts["topology"]
    types = topology["operation_types"]
    width, height = topology["width"], topology["height"]
    jobs = {r["id"]: r for r in facts["jobs"]}

    def summary(ids):
        rows = [jobs[j] for j in ids if j in jobs]
        if not rows:
            return (0.0,) * 8
        slack = [(r["due_at"] - facts["tick"]) / t for r in rows]
        return (
            len(rows) / n,
            min(slack),
            sum(slack) / len(rows),
            sum(x < 0 for x in slack) / n,
            sum(r["rush"] for r in rows) / n,
            sum(r["step"] for r in rows) / n,
            sum(len(r["remaining_operations"]) for r in rows) / n,
            sum(r["public_risk"] for r in rows) / len(rows),
        )

    state = [facts["tick"] / t, len(facts["completed"]) / n, *summary(jobs)]
    for row in machines:
        state.extend(
            (
                float(row["down"]),
                float(row["status"] == "PROCESSING"),
                float(row["status"] == "BLOCKED"),
                float(row["job"] is not None),
                row["elapsed"] / t,
                row["nominal"] / t,
                *one_hot(row["mode"], ("slow", "normal", "fast")),
                *(float(k in row["capabilities"]) for k in types),
                *summary((row["job"],)),
            )
        )
    for row in vehicles:
        target = row["target"]
        state.extend(
            (
                row["cell"][0] / width,
                row["cell"][1] / height,
                float(row["job"] is not None),
                float(row["service"] is not None),
                (facts["tick"] - row["arrived_at"]) / t,
                *one_hot(target["owner"] if target else None, owners),
                *summary((row["job"],)),
            )
        )
    for row in buffers:
        capacities = tuple(row["capacity"].values())
        inventory = [j for values in row["inventory"].values() for j in values]
        state.extend(
            (
                sum(c or 0 for c in capacities) / n,
                float(any(c is None for c in capacities)),
                row["quota"] / n,
                len(row["commitments"]) / n,
                len(row["ready"]) / n,
                *summary(inventory),
            )
        )
    solids = set(topology["solids"])
    state.extend(float((x, y) in solids) for y in range(height) for x in range(width))
    for owner in owners:
        cells = topology["ports"].get(owner, ())
        state.extend(
            (
                len(cells) / n,
                sum(c[0] / width for c in cells) / max(1, len(cells)),
                sum(c[1] / height for c in cells) / max(1, len(cells)),
            )
        )
    layout = (
        owners,
        types,
        plain(topology),
        tuple((r["id"], r["capabilities"]) for r in machines),
    )
    signature = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return tuple(state), owners, signature
