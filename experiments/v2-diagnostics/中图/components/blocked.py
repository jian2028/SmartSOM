"""Restrict pickup rule ranking to ready outputs of blocked machines, if any.

This is a dispatch heuristic, not a physical feasibility or deadlock guarantee.
All original pickup rules rank the same filtered legal source set.
"""

from smartsom.representations.v2.rules import choose
from smartsom.representations.v2.rules import create_rule as original_factory


def create_rule(*, rule):
    original_factory(rule=rule)  # Validate the unchanged named rule catalog.

    def select(facts, request, candidates):
        if request.group == "dispatcher" and request.stage == "pickup":
            blocked = tuple(
                c
                for c in candidates
                if facts.roles[c.action[0]] == "machine_post"
                and facts.machine_state[facts.station[c.action[0]]]["status"]
                == "BLOCKED"
                and facts.ready(c.action[0])
            )
            if blocked:
                candidates = blocked
        return choose(facts, request, candidates, rule)

    return select


create_rule.CONTRACT = "smartsom.rule/v1"
