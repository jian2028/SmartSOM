"""Candidate-linked public pressure context, a separate observation ablation."""

from dataclasses import replace

import torch

from smartsom.representations.v2 import network, observation
from smartsom.representations.v2.facts import view
from smartsom.representations.v2.rules import CATALOG

SCHEMA = "smartsom.v2-diagnostic.pressure/v1"
EXTRA = 4


def create_observation(*, social_enabled=False):
    original = observation.create_observation(social_enabled=social_enabled)

    def observe(facts, request):
        result = original(facts, request)
        if request.group == "dispatcher" and request.stage == "pickup":
            v = view(facts)
            posts = tuple(facts.post.values())
            machines = tuple(facts.machine_state.values())
            agvs = tuple(facts.agvs.values())
            pressure = (
                sum(s["status"] == "BLOCKED" for s in machines) / len(machines),
                sum(v.occupancy(p) for p in posts) / len(posts),
                sum(bool(facts.ready(p)) and not facts.commitments(p) for p in posts)
                / len(posts),
                sum(s["job"] is not None for s in agvs) / len(agvs),
            )
            result["candidates"] = tuple(
                (*row, *pressure) for row in result["candidates"]
            )
        return result

    return observe


def create_network(*, group, options, value_head, social_enabled=False):
    result = network.create_network(
        group=group,
        options=options,
        value_head=value_head,
        social_enabled=social_enabled,
    )
    if group == "dispatcher.pickup":
        width = observation.FEATURES[group] + len(CATALOG[group]) + EXTRA
        result.branches["pickup"].candidate = network.mlp(
            "diagnostic-pressure/candidate", [width, 32, 32]
        )
    return result


def pack(requests, *, critics, options):
    stripped = [
        replace(
            r,
            inputs={
                **r.inputs,
                "candidates": tuple(row[:-EXTRA] for row in r.inputs["candidates"]),
            },
        )
        if r.group == "dispatcher" and r.stage == "pickup"
        else r
        for r in requests
    ]
    result = network.pack(stripped, critics=critics, options=options)
    if requests[0].group == "dispatcher" and "pickup" in result["stages"]:
        data = result["stages"]["pickup"]
        extra = torch.zeros(*data["candidates"].shape[:2], EXTRA)
        for i, index in enumerate(data["indices"].tolist()):
            rows = requests[index].inputs["candidates"]
            extra[i, : len(rows)] = torch.tensor([row[-EXTRA:] for row in rows])
        data["candidates"] = torch.cat((data["candidates"], extra), dim=-1)
    return result


create_observation.OUTPUT_SCHEMA = SCHEMA
create_network.INPUT_SCHEMA = SCHEMA
create_network.pack = pack
