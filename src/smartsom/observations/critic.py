"""Default value inputs, explicitly selected independently from Actor inputs."""

from smartsom.observations.base import Projection


def create_observation():
    def observe(facts, context):
        return Projection(facts, context).critic_features()

    observe.subject_independent = True
    return observe


create_observation.OUTPUT_SCHEMA = "smartsom.critic/base48-v1"
