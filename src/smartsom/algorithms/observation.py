"""One component boundary: public facts and semantic candidates -> model inputs."""

from dataclasses import replace

from smartsom.config.components import construct, construct_observation
from smartsom.domain.public_facts import freeze


def build_observation(options):
    encoders = {
        name: construct_observation(policy.observation)
        for name, policy in options.production.policies.items()
    }
    representation = options.action_representation
    custom = {
        name: construct(spec) for name, spec in representation.custom_rules.items()
    }

    def project(facts, decision):
        point = (
            "central.assign"
            if options.control == "centralized"
            else decision.group + "." + decision.stage
        )
        if options.control == "base" and representation.mode(point) == "rules":
            from smartsom.algorithms.rule_actions import project as project_rules
            from smartsom.observations.rules import candidate_facts

            rules = representation.selected_rules(point)
            values = (
                {}
                if rules and all(rule in custom for rule in rules)
                else candidate_facts(facts, decision)
            )
            decision, _ = project_rules(
                decision,
                values,
                rules=rules,
                custom=custom,
                facts=facts,
                merge_aliases=representation.merge_aliases,
            )
        name = options.production.bindings.get(point, options.production.default)
        return replace(decision, inputs=freeze(encoders[name](facts, decision)))

    return project
