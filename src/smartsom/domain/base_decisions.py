"""Detached, semantic decisions for reservation-based self-organizing control."""

from dataclasses import dataclass

from smartsom.domain.social import SocialCandidate

BASE_CONTRACT = "smartsom.base-reservations/v1"
GROUPS = ("machine", "dispatcher", "pre", "post", "input")
FEATURES = 48
SET_NAMES = ("jobs", "applicants", "commitments", "stations")


@dataclass(frozen=True, slots=True)
class BaseCandidate:
    identity: str
    action: tuple[str, ...]
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BaseDecision:
    tick: int
    group: str
    owner: str
    stage: str
    candidates: tuple[BaseCandidate, ...]
    context: tuple[float, ...]
    sets: tuple[tuple[tuple[float, ...], ...], ...]
    rejected_owners: tuple[str, ...] = ()
    social: tuple[SocialCandidate, ...] | None = None
    inputs: object = None


def vector(*values):
    if len(values) > FEATURES:
        raise ValueError("feature schema overflow")
    return tuple(float(x) for x in values) + (0.0,) * (FEATURES - len(values))
