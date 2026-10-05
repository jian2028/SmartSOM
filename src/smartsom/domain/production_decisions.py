"""Framework-free semantic decisions for the composable production contract."""

from dataclasses import dataclass
from typing import Literal

ROLE_NAMES = ("machine", "buffer", "dispatcher", "mover")
MOVEMENT_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
ACTION_CONTRACT = "smartsom.production-actions/v3.1"
OBSERVATION_CONTRACT = "smartsom.production-observations/v3.1"


@dataclass(frozen=True, slots=True)
class Candidate:
    identity: str
    action: object
    features: tuple[float, ...]
    legal: bool = True


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    tick: int
    stage: str
    role: Literal["machine", "buffer", "dispatcher", "mover"]
    owner: str
    candidates: tuple[Candidate, ...]
    observation: dict
    prefix: tuple[str, ...] = ()
    count: int = 1

    @property
    def identity(self):
        return f"{self.tick}:{self.stage}:{self.role}:{self.owner}"

    @property
    def deterministic(self):
        return sum(c.legal for c in self.candidates) <= 1


@dataclass(frozen=True, slots=True)
class DispatchTarget:
    owner: str
    port: str


@dataclass(frozen=True, slots=True)
class BoundaryCommand:
    """A complete replayable semantic boundary, never candidate array indices."""

    machines: tuple = ()
    dispatchers: tuple = ()
    prefixes: tuple = ()
    matching: tuple = ()
    movers: tuple = ()
    contract: str = ACTION_CONTRACT

    def __post_init__(self):
        if self.contract != ACTION_CONTRACT:
            raise ValueError("incompatible physical decision contract")
        for name in ("machines", "dispatchers", "prefixes", "movers"):
            rows = tuple(getattr(self, name))
            if len({key for key, _ in rows}) != len(rows):
                raise ValueError(f"duplicate owner in {name}")
            object.__setattr__(self, name, rows)
        object.__setattr__(self, "matching", tuple(self.matching))
