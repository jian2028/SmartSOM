"""Framework-independent learning contracts; optional backends import separately."""

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
