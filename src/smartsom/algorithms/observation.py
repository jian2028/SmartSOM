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
    representation = options.action_representation
    custom = {
        name: construct(spec) for name, spec in representation.custom_rules.items()
    }
    from smartsom.algorithms.base_policy import DECISION_POINTS

    rule_points = {p for p in DECISION_POINTS if representation.mode(p) == "rules"}
    if not encoders and not rule_points:
        return None

    def name_for(request):
        point = (
            "central.assign"
            if options.control == "centralized"
            else request.group + "." + request.stage
        )
        return options.production.bindings.get(point, options.production.default)

    def project(facts, decision):
        point = decision.group + "." + decision.stage
        if point in rule_points:
            from smartsom.algorithms.rule_actions import project as project_rules

            decision, _ = project_rules(
                decision,
                rules=representation.selected_rules(point),
                custom=custom,
                facts=facts,
            )
        encoder = encoders.get(name_for(decision))
        return (
            replace(decision, inputs=freeze(encoder(facts, decision)))
            if encoder
            else decision
        )

    project.needs_facts = lambda request: (
        name_for(request) in encoders
        or bool(
            set(
                (
                    representation.selected_rules(request.group + "." + request.stage)
                    or ()
                )
            )
            & custom.keys()
        )
    )
    project.applies = lambda request: (
        name_for(request) in encoders
        or request.group + "." + request.stage in rule_points
    )
    project.needs_topology = lambda request: bool(
        getattr(encoders.get(name_for(request)), "needs_topology", False)
    )
    return project
