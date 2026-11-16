"""Explicit fixed pickup baseline: HPOF when available, otherwise NPF."""

from smartsom.representations.v2.rules import choose


def create_rule(*, rule="HPOF"):
    if rule != "HPOF":
        raise ValueError("this composite pickup baseline only replaces HPOF")

    def select(facts, request, candidates):
        if request.group != "dispatcher" or request.stage != "pickup":
            raise ValueError("HPOF/NPF baseline requires dispatcher.pickup")
        primary = choose(facts, request, candidates, "HPOF")
        return (
            primary
            if primary is not None
            else choose(facts, request, candidates, "NPF")
        )

    return select


create_rule.CONTRACT = "smartsom.rule/v1"
