"""Startup-bound per-policy projections; candidates remain protocol-owned."""

from dataclasses import replace

from smartsom.config.components import construct
from smartsom.domain.public_facts import freeze


def build_observation(options):
    encoders = {
        name: construct(policy.observation)
        for name, policy in options.production.policies.items()
        if policy.observation is not None
    }
    encoders = {
        name: encoder for name, encoder in encoders.items() if encoder is not None
    }
    if not encoders:
        return None

    def name_for(request):
        point = (
            "central.assign"
            if options.control == "centralized"
            else request.group + "." + request.stage
        )
        return options.production.bindings.get(point, options.production.default)

    def project(facts, decision):
        encoder = encoders.get(name_for(decision))
        return (
            replace(decision, inputs=freeze(encoder(facts, decision)))
            if encoder
            else decision
        )

    project.needs_facts = lambda request: name_for(request) in encoders
    return project
