"""Portable semantic request records for rollout transfer and replay."""

from dataclasses import asdict

from smartsom.domain.base_decisions import BaseCandidate, BaseDecision


def request_from_dict(data):
    data = dict(data)
    data["candidates"] = tuple(
        BaseCandidate(c["identity"], tuple(c["action"]), tuple(c["features"]))
        for c in data["candidates"]
    )
    if data.get("social") is not None:
        from smartsom.domain.social import SocialCandidate, SocialLink

        data["social"] = tuple(
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
            for c in data["social"]
        )
    if data.get("inputs") is not None:
        from smartsom.domain.public_facts import freeze

        data["inputs"] = freeze(data["inputs"])
    return BaseDecision(**data)


def recorded_request(request):
    from smartsom.domain.public_facts import plain

    result = asdict(request)
    result["inputs"] = plain(request.inputs)
    return result
