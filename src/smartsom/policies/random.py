"""Uniform legal-candidate production policy, independent of traffic."""


def create_policy(*, seed=0, control="base"):
    if control == "centralized":
        from smartsom.algorithms.central_rules import CentralPolicy

        return CentralPolicy("random", seed)
    from smartsom.policies.rule import BaseRule

    return BaseRule("random", seed)


create_policy.CONTRACT = "smartsom.policy/v1"
