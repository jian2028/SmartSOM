"""Current local Base features constructed from public facts by a component."""

from smartsom.observations.base import Projection


def create_observation():
    def observe(facts, decision):
        return Projection(facts, decision).build()

    return observe


create_observation.OUTPUT_SCHEMA = "smartsom.base-observation/native-v2"
