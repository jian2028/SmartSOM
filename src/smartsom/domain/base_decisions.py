"""Detached, semantic decisions for reservation-based self-organizing control."""

from dataclasses import KW_ONLY, dataclass, field

BASE_CONTRACT = "smartsom.base-reservations/v1"
GROUPS = ("machine", "dispatcher", "pre", "post", "input")


@dataclass(frozen=True, slots=True)
class BaseCandidate:
    identity: str
    action: tuple[str, ...]
    rule_code: int = field(default=0, kw_only=True)


@dataclass(frozen=True, slots=True)
class BaseDecision:
    tick: int
    group: str
    owner: str
    stage: str
    candidates: tuple[BaseCandidate, ...]
    _: KW_ONLY
    rejected_owners: tuple[str, ...] = ()
    inputs: object = None
    rule_audit: object = None
