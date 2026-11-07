"""Immutable output of the current Base/SL observation component."""

from dataclasses import dataclass

from smartsom.domain.social import SocialCandidate


@dataclass(frozen=True, slots=True)
class BaseInputs:
    context: tuple
    sets: tuple
    candidates: tuple
    social: tuple[SocialCandidate, ...] | None = None


FEATURES = 48
SET_NAMES = ("jobs", "applicants", "commitments", "stations")


def vector(*values):
    if len(values) > FEATURES:
        raise ValueError("feature schema overflow")
    return tuple(float(x) for x in values) + (0.0,) * (FEATURES - len(values))
