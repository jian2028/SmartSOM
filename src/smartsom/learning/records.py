"""Portable semantic requests and explicitly typed observation outputs."""

from dataclasses import asdict

from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
from smartsom.domain.observation_data import BaseInputs
from smartsom.domain.public_facts import freeze, plain
from smartsom.domain.social import SocialCandidate, SocialLink


def request_from_dict(data):
    data = dict(data)
    kind = data.pop("input_type")
    data["candidates"] = tuple(
        BaseCandidate(c["identity"], tuple(c["action"]), rule_code=c["rule_code"])
        for c in data["candidates"]
    )
    data["rejected_owners"] = tuple(data["rejected_owners"])
    inputs = data["inputs"]
    if kind == "base":
        social = (
            None
            if inputs["social"] is None
            else tuple(
                SocialCandidate(
                    c["identity"],
                    tuple(
                        SocialLink(
                            link["identity"],
                            link["kind"],
                            tuple(link["query"]),
                            tuple(tuple(row) for row in link["peers"]),
                            link["terminal"],
                            link["count"],
                        )
                        for link in c["links"]
                    ),
                )
                for c in inputs["social"]
            )
        )
        data["inputs"] = BaseInputs(
            tuple(inputs["context"]),
            tuple(tuple(tuple(row) for row in rows) for rows in inputs["sets"]),
            tuple(tuple(row) for row in inputs["candidates"]),
            social,
        )
    elif kind == "named":
        data["inputs"] = freeze(inputs)
    else:
        raise ValueError("unknown observation record type")
    return BaseDecision(**data)


def recorded_request(request):
    result = asdict(request)
    result["input_type"] = "base" if isinstance(request.inputs, BaseInputs) else "named"
    if result["input_type"] == "named":
        result["inputs"] = plain(request.inputs)
    return result
