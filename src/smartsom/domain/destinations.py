"""Semantic destinations shared by movement actions and resources."""

from dataclasses import dataclass
from typing import Literal

from smartsom.domain.validation import DomainValidationError, _identifier


@dataclass(frozen=True, slots=True)
class TransportDestination:
    kind: Literal["machine", "output"]
    machine_id: str | None = None

    def __post_init__(self):
        if self.kind == "machine":
            _identifier(self.machine_id, "destination machine_id")
        elif self.kind != "output" or self.machine_id is not None:
            raise DomainValidationError("destination must be a machine or output")
