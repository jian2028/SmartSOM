"""Build explicit numerical model-input fixtures independently of the engine."""

from dataclasses import dataclass

from smartsom.domain.base_decisions import BaseCandidate, BaseDecision
from smartsom.domain.observation_data import BaseInputs


@dataclass(frozen=True)
class NumericRow:
    identity: str
    action: tuple
    features: tuple


def numeric_candidate(identity, action, features):
    return NumericRow(identity, action, features)


def numeric_request(
    tick,
    group,
    owner,
    stage,
    rows,
    context,
    sets,
    rejected_owners=(),
    social=None,
    inputs=None,
    rule_audit=None,
):
    rows = tuple(rows)
    return BaseDecision(
        tick,
        group,
        owner,
        stage,
        tuple(BaseCandidate(r.identity, r.action) for r in rows),
        rejected_owners=rejected_owners,
        inputs=inputs
        if inputs is not None
        else BaseInputs(context, sets, tuple(r.features for r in rows), social),
        rule_audit=rule_audit,
    )


def observe(protocol, request):
    from smartsom.engine.decision_facts import DecisionFacts
    from smartsom.observations.base import Projection

    if request.inputs is not None:
        return request
    with protocol.local_facts(), DecisionFacts(protocol, request) as facts:
        from dataclasses import replace

        return replace(
            request, inputs=Projection(facts, request, arm=protocol.social_arm).build()
        )


def rule_action(protocol, request):
    from dataclasses import replace

    from smartsom.domain.public_facts import freeze
    from smartsom.engine.decision_facts import DecisionFacts
    from smartsom.observations.rules import create_observation
    from smartsom.policies.rule import BaseRule

    with protocol.local_facts(), DecisionFacts(protocol, request) as facts:
        return BaseRule().choose(
            replace(request, inputs=freeze(create_observation()(facts, request)))
        )


def feature_query(protocol, name, *args):
    from smartsom.engine.decision_facts import DecisionFacts
    from smartsom.observations.base import Projection

    request = BaseDecision(protocol.core.tick, "value", "factory", "value", ())
    with protocol.local_facts(), DecisionFacts(protocol, request) as facts:
        if name == "social":
            from smartsom.observations.social import project_social

            return project_social(Projection(facts, request), *args)
        return getattr(Projection(facts, request), name)(*args)
