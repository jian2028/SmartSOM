"""Base operational inputs plus explicit candidate-associated peer context."""

from smartsom.domain.social import SOCIAL_ARMS
from smartsom.observations.base import Projection


def create_observation(*, arm="all"):
    if arm not in SOCIAL_ARMS:
        raise ValueError("unknown social role/stage arm")

    def observe(facts, decision):
        return Projection(facts, decision, arm=arm).build()

    return observe


create_observation.OUTPUT_SCHEMA = "smartsom.social-observation/v1"
