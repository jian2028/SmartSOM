"""Framework-independent learning contracts; optional backends import separately."""

# Registers built-in reward shaping without changing extensions.py source digests.
import smartsom.learning.production_shaping  # noqa: F401
import smartsom.learning.reward_shaping  # noqa: F401
from smartsom.learning.projection import (
    LearningProjection,
    ProjectedDecision,
    ProjectionSpec,
    validate_capacity,
)

__all__ = [
    "LearningProjection",
    "ProjectionSpec",
    "ProjectedDecision",
    "validate_capacity",
]
