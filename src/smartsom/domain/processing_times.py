"""Immutable realized mode durations, separate from nominal workload content."""

from dataclasses import dataclass

from smartsom.domain.models import WorkloadInstance, _identifier, _items


@dataclass(frozen=True, slots=True)
class ProcessingTime:
    operation_id: str
    processing_mode_id: str
    nominal_ticks: int
    actual_ticks: int

    def __post_init__(self):
        _identifier(self.operation_id, "operation_id")
        _identifier(self.processing_mode_id, "processing_mode_id")
        if any(
            type(value) is not int or value < 1
            for value in (self.nominal_ticks, self.actual_ticks)
        ):
            raise ValueError("nominal_ticks and actual_ticks must be positive integers")


@dataclass(frozen=True, slots=True)
class ProcessingTimePlan:
    modes: tuple[ProcessingTime, ...]

    def __post_init__(self):
        entries = _items(self.modes, ProcessingTime, "processing times")
        keys = [(row.operation_id, row.processing_mode_id) for row in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate processing time identity")
        object.__setattr__(
            self,
            "modes",
            tuple(
                sorted(
                    entries, key=lambda row: (row.operation_id, row.processing_mode_id)
                )
            ),
        )

    def validate(self, workload: WorkloadInstance) -> None:
        expected = {
            (op.operation_id, mode.processing_mode_id): mode.nominal_ticks
            for op in workload.operations
            for mode in op.modes
        }
        actual = {
            (row.operation_id, row.processing_mode_id): row.nominal_ticks
            for row in self.modes
        }
        if expected.keys() != actual.keys():
            raise ValueError(
                f"processing time coverage mismatch: missing={sorted(expected.keys() - actual.keys())}, unknown={sorted(actual.keys() - expected.keys())}"
            )
        if expected != actual:
            raise ValueError("processing time nominal_ticks disagree with workload")
