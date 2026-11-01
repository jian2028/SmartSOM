"""Peer OFF removes candidate-linked peer inputs, retaining the same network."""

from dataclasses import replace

from smartsom.domain.base_decisions import vector


def project(request, *, peer_enabled):
    if peer_enabled or (request.group, request.stage) != ("dispatcher", "pickup"):
        return request
    if request.social is None:
        raise ValueError("pickup intervention requires social observations")
    return replace(
        request,
        social=tuple(
            replace(
                candidate,
                links=tuple(
                    replace(link, peers=(), query=vector(), count=0, terminal=False)
                    for link in candidate.links
                ),
            )
            for candidate in request.social
        ),
    )
