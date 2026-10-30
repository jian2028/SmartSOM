"""Explicit social observation selection over unchanged Base operational inputs."""

from smartsom.domain.social import SOCIAL_ARMS
from smartsom.representations.base.observation import (
    create_observation as base_observation,
)


def create_observation(*, arm="all"):
    if arm != "base" and arm not in SOCIAL_ARMS:
        raise ValueError("unknown social role/stage arm")
    return base_observation()


create_observation.OUTPUT_SCHEMA = "smartsom.social-observation/v1"
