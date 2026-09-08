"""Semantic destinations shared by movement actions and resources."""

from dataclasses import dataclass
from typing import Literal

from smartsom.domain.validation import DomainValidationError, _identifier


@dataclass(frozen=True, slots=True)
class TransportDestination:
    kind: Literal["machine", "output", "holding"]
    machine_id: str | None = None
    buffer_id: str | None = None

    def __post_init__(self):
        if self.kind == "machine":
            _identifier(self.machine_id, "destination machine_id")
            if self.buffer_id is not None:
                raise DomainValidationError("machine destination cannot have buffer_id")
        elif self.kind == "holding":
            _identifier(self.buffer_id, "destination buffer_id")
            if self.machine_id is not None:
                raise DomainValidationError(
                    "holding destination cannot have machine_id"
                )
        elif (
            self.kind != "output"
            or self.machine_id is not None
            or self.buffer_id is not None
        ):
            raise DomainValidationError(
                "destination must be a machine, holding buffer or output"
            )
